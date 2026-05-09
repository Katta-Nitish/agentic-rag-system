import arxiv
import os

os.makedirs("arxiv_corpus", exist_ok=True)

client = arxiv.Client()
search = arxiv.Search(
  query = "cat:cs.AI",
  max_results = 50,
  sort_by = arxiv.SortCriterion.SubmittedDate
)

print("Downloading...")

for i, result in enumerate(search.results()):
    filename = f"{result.get_short_id().replace('/', '_')}.pdf"
    filepath = os.path.join("arxiv_corpus", filename)
    
    print(f"Downloading {i+1}/50: {result.title}")
    result.download_pdf(dirpath="arxiv_corpus", filename=filename)

print("Download complete!.")