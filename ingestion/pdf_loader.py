from unstructured.partition.pdf import partition_pdf

def load_pdf(pdf_path, image_dir):
    return partition_pdf(
        filename=pdf_path,
        extract_images_in_pdf=True,
        extract_image_block_output_dir=image_dir,
        infer_table_structure=True,
        chunking_strategy="by_title",
        max_characters=4000,
    )
