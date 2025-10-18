import sys
import requests
import datetime
import random
import json
from io import BytesIO
from PIL import Image
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from better_profanity import profanity

current_time = datetime.datetime.now()
Year = current_time.year
Month = current_time.month
Day = current_time.day - 2
    
BaseLink = "https://en.wikipedia.org/w/api.php"
PopularArticlesAPI = f"https://wikimedia.org/api/rest_v1/metrics/pageviews/top/en.wikipedia/all-access/{Year}/{Month}/{Day}" # Used so the articles make sense and arent extremely niche

UserAgent = "WikiQuizBot/0.1 (contact: wikiquizbotcontactemail.unified438@slmails.com)"
DefaultHeaders = {"User-Agent":UserAgent}



def mcq_array(title: str):

    print(title)
    model_name = "Qwen/Qwen2.5-1.5B-Instruct"

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map="auto",
        dtype="auto"
    )

    gen = pipeline("text-generation", model=model, tokenizer=tokenizer)

    prompt = f"""You are a helpful assistant. I will give you a correct answer, and you will generate 3 plausible but incorrect distractor options related to it.

Return your output as a JSON array of strings, **only including the 3 distractors**, no explanation.

Correct answer: "{title}"

Output:


"""

    out = gen(prompt, max_new_tokens=128, temperature=0.7, do_sample=True, return_full_text=False)[0]["generated_text"]
    out = out.strip()
    
    try:
        arr = json.loads(out)
        return arr
    except:
        print(f"This topic: {title} failed. the generated array is: {out}")
        return None
    


def IdentifyNSFWText(string):
    string = string.lower()
    if "sex" in string:
        return True
    if "porn" in string:
        return True
    if (profanity.contains_profanity(string)):
        return True
    
    return False


r = requests.get(PopularArticleAPI, headers=DefaultHeaders)
data = r.json()


articles = data["items"][0]["articles"]

filtered = [
    page for page in articles
    if not (page["article"] == "Main_Page" or ":" in page["article"])
]

randomArticle = random.sample(filtered, 50)

PagesSelected = []

for i, article in enumerate(randomArticle, 1):
    
    ArticleName = article["article"]
    
    print(i)

    params = {
        "action": "query",
        "titles": ArticleName,
        "prop": "pageimages",
        "format": "json",
        "pithumbsize": 1080
    }
    
    r = requests.get(BaseLink, headers=DefaultHeaders, params=params)
    data = r.json()

    Pages = data["query"]["pages"]
    page = next(iter(Pages.values()))


    if "thumbnail" in page:
        PagesSelected.append(page)


print(PagesSelected)

for page in PagesSelected:
    thumbnailURL = page["thumbnail"]["source"]
    PageTitle = page['title']

    print(thumbnailURL)

    # nsfw image detection

    classifier = pipeline("image-classification", model="Falconsai/nsfw_image_detection")
    r = requests.get(thumbnailURL, stream=True, headers=DefaultHeaders)
    r.raise_for_status()
    
    img = Image.open(BytesIO(r.content))
    result = classifier(img)
    NormalLabel = result[0] 
    NSFWLabel = result[1]

    NSFWText = IdentifyNSFWText(PageTitle)   

    if NSFWText:
        print("bad boy")
        print(NSFWText, PageTitle)
        PagesSelected.remove(page)
        continue
   
    if NSFWLabel['score'] >= 0.0005:
        print("AHHHH")
        PagesSelected.remove(page)
    
info = {}

print(PagesSelected)

PagesSelected = PagesSelected[:20]

for filteredPages in PagesSelected:
    thumbnailURL = filteredPages["thumbnail"]["source"]
    PageTitle = filteredPages['title']

    print(thumbnailURL)
    result = mcq_array(PageTitle)
    if result == None:
        continue
    info[PageTitle] = result

    if (len(info)) >= 10:
        break

print(info, len(info))
    
   


