#!/usr/bin/env python3
import argparse
import warnings
from transformers import pipeline, AutoModelForCausalLM, AutoTokenizer
from pathlib import Path

warnings.filterwarnings("ignore")


def parse_input() -> tuple[str, Path]:
    """
    Парсит --prompt и --output
    Возвращает: (keywords, output_path)
    """
    parser = argparse.ArgumentParser(
        description="Генерация статьи c HTML разметкой"
    )
    parser.add_argument(
        "--prompt",
        type=str,
        required=True,
        help="5 слов через пробел, например: волна корабль плыть приключение сокровища"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="article.html",
        help="Имя выходного HTML (по умолчанию: article.html)"
    )

    args = parser.parse_args()

    # Валидация 5 слов
    words = args.prompt.strip().split()
    if len(words) != 5:
        raise ValueError(
            f"Ошибка: нужно ровно 5 слов, получено {len(words)}.\n"
            f"Пример: --prompt 'волна корабль плыть приключение сокровища'"
        )

    keywords = " ".join(words)
    output_path = Path(args.output)

    return keywords, output_path


def generate_raw_text(keywords: str) -> str:
    print(f"Генерация статьи по теме: {keywords}\n")

    model_name = "Qwen/Qwen2.5-Coder-7B-Instruct"

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map="auto",
        torch_dtype="auto"
    )

    pipe = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        device_map="auto"
    )

    prompt = (
        f"Напиши информативную и логичную статью на русском языке используя все слова из: {keywords}. "
        f"Структурируй текст обособленными #заголовком, #абзацами и #выводом. Пиши только текст статьи."
    )

    result = pipe(
        prompt,
        max_new_tokens=800,
        temperature=0.8,
        top_p=0.9,
        do_sample=True
    )

    return result[0]["generated_text"]


def clean_and_parse_qwen_output(raw_text: str) -> dict:
    """
    Убирает промпт, парсит # заголовки.
    ВАЖНО: Текст после #Вывод полностью сохраняется!
    """
    lines = raw_text.strip().split('\n')
    parsed = {
        'title': '',
        'sections': [],
        'conclusion': ''
    }

    # Отрезаем всё до первого #
    try:
        first_hash = next(i for i, line in enumerate(lines) if line.strip().startswith('#'))
        lines = lines[first_hash:]
    except StopIteration:
        lines = []

    current_h2 = None
    current_p_lines = []
    in_conclusion = False  #Флаг: идём ли в выводе

    for line in lines:
        line = line.strip()
        if not line:
            continue

        if line.startswith('#'):
            # Сохраняем предыдущий блок
            if current_h2 and current_p_lines:
                if in_conclusion:
                    parsed['conclusion'] = ' '.join(current_p_lines).strip()
                else:
                    parsed['sections'].append({
                        'h2': current_h2,
                        'p': ' '.join(current_p_lines).strip()
                    })
                current_p_lines = []

            # Новый заголовок
            header = line[1:].strip()
            if not parsed['title']:
                parsed['title'] = header
            elif header.lower().startswith("вывод"):
                in_conclusion = True
                current_h2 = "Вывод"
            else:
                current_h2 = header
                in_conclusion = False
        else:
            # Текст абзаца
            if current_h2:
                current_p_lines.append(line)

    # Последний блок (после #Вывод)
    if current_h2 and current_p_lines:
        if in_conclusion:
            parsed['conclusion'] = ' '.join(current_p_lines).strip()
        else:
            parsed['sections'].append({
                'h2': current_h2,
                'p': ' '.join(current_p_lines).strip()
            })

    return parsed


def generate_html(parsed: dict, output_path: Path):
    """
    Создаёт красивый HTML с CSS
    """
    html = f"""<!DOCTYPE html>
<html lang="ru">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{parsed['title']}</title>
    <style>
        body {{ font-family: 'Segoe UI', sans-serif; line-height: 1.7; max-width: 800px; margin: 40px auto; padding: 20px; background: #f9f9fb; color: #333; }}
        h1 {{ color: #2c3e50; border-bottom: 3px solid #3498db; padding-bottom: 10px; }}
        h2 {{ color: #2980b9; margin-top: 30px; }}
        p {{ margin: 15px 0; text-align: justify; }}
        .conclusion {{ background: #ecf0f1; padding: 20px; border-left: 5px solid #3498db; border-radius: 5px; }}
    </style>
</head>
<body>
    <h1>{parsed['title']}</h1>
"""

    for section in parsed['sections']:
        html += f"    <h2>{section['h2']}</h2>\n"
        html += f"    <p>{section['p']}</p>\n"

    if parsed['conclusion']:
        html += f"    <div class='conclusion'>\n"
        html += f"        <h2>Вывод</h2>\n"
        conclusion_text = parsed['conclusion'].replace("вывод:", "", 1).strip()
        html += f"        <p>{conclusion_text}</p>\n"
        html += f"    </div>\n"

    html += """
</body>
</html>"""

    output_path.write_text(html, encoding='utf-8')
    print(f"\nHTML сохранён: {output_path}")


def main():
    try:
        keywords, output_path = parse_input()
    except ValueError as e:
        print(e)
        return

    raw_text = generate_raw_text(keywords)
    parsed = clean_and_parse_qwen_output(raw_text)
    generate_html(parsed, output_path)


if __name__ == "__main__":
    main()