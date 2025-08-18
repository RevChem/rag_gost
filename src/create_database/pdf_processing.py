import pdfplumber
import pandas as pd
from pdfplumber.utils import extract_text, get_bbox_overlap, obj_to_bbox

def process_pdf(pdf_path):
    pdf = pdfplumber.open(pdf_path)
    all_text = []

    for page in pdf.pages:
        page = page.crop(page.bbox)
        page_text_full = page.extract_text()

        # Игнорирование титульной страницы и содержания. 
        if "Издание официальное" in (page_text_full or "") or 'Содержание' in (page_text_full or ""):
            continue  

        filtered_page = page
        chars = filtered_page.chars

        for table in page.find_tables():
            table_data = table.extract()
            table_data = [[cell if cell is not None else "" for cell in row] for row in table_data]

            df = pd.DataFrame(table_data)
            if len(df) == 0:
                continue

            df.columns = df.iloc[0]
            markdown = df.drop(0).to_markdown(index=False)

            filtered_page = filtered_page.filter(lambda obj:
                get_bbox_overlap(obj_to_bbox(obj), table.bbox) is None
            )
            chars = filtered_page.chars

            first_table_char = page.crop(table.bbox).chars[0]
            chars.append({
                **first_table_char,
                "text": markdown
            })

        page_text = extract_text(chars, layout=True)

        if "Библиография" in page_text:
            page_text = page_text[: page_text.find("Библиография")].strip()

        if "Редактор" in page_text:
            page_text = page_text[: page_text.find("Редактор")].strip()

        all_text.append(page_text)

    pdf.close()
    return "\n".join(all_text)