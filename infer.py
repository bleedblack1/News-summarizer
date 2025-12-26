from transformers import BartTokenizerFast, BartForConditionalGeneration
import torch

MODEL_DIR = "./bart-news-finetuned"

tokenizer = BartTokenizerFast.from_pretrained(MODEL_DIR)
model = BartForConditionalGeneration.from_pretrained(MODEL_DIR).cuda()
model.eval()

def summarize(text):
    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=768
    ).to("cuda")

    with torch.no_grad():
        summary_ids = model.generate(
            **inputs,
            max_length=128,
            num_beams=4,
            early_stopping=True
        )

    return tokenizer.decode(summary_ids[0], skip_special_tokens=True)

if __name__ == "__main__":
    article = """
    The United States signed the International Convention on the Elimination of All Forms of Racial Discrimination (“ICERD” or “Convention”) in 1966. President Lyndon Johnson’s administration noted at the time that the United States “has not always measured up to its constitutional heritage of equality for all” but that it was “on the march” toward compliance.[1] The United States finally ratified the Convention in 1994 and first reported on its progress in implementing the Convention to the Committee on the Elimination of Racial Discrimination (“CERD” or “Committee”) in 2000. In August 2022, the Committee will examine the combined 10th – 12th periodic reports by the United States on compliance with the Convention. This report supplements the submission of the government with additional information in key areas and offers recommendations that will, if adopted, enhance the government’s ability to comply with ICERD.

In its 2000 report, the United States stated that “overt discrimination” is “less pervasive than it was thirty years ago” but admitted it continued due to “subtle forms of discrimination” that “persist[ed] in American society.”[2] The forms of discrimination reported to the United Nations by the United States included “inadequate enforcement of existing anti-discrimination laws”; “ineffective use and dissemination of data”; economic disadvantage experienced by minority groups; “persistent discrimination in employment and labour relations”; “segregation and discrimination in housing” leading to diminished educational opportunities for minorities; lack of equal access to capital, credit markets and technology; discrimination in the criminal legal system; lack of adequate access to health insurance and health care; and discrimination against immigrants, among other harmful effects.[3] The United States also noted the heightened impact of racism on women and children.

It has now been more than 50 years since the US signed the ICERD, nearly 30 years since it ratified the Convention, and more than 20 since it identified the extensive foregoing list of impediments to its effective implementation. Yet progress toward compliance remains elusive—indeed, grossly inadequate—in numerous key areas including reparative justice; discrimination in the US criminal legal system; use of force by law enforcement officials; discrimination in the regulation and enforcement of migration control; and stark disparities in the areas of economic opportunity and health care. Structural racism and xenophobia persist as powerful and pervasive forces in American society.

During his first days in office, President Joseph Biden called for urgent action to advance equity for all, calling this a “battle for the soul of [the] nation” because “systemic racism” is “corrosive,” “destructive” and “costly.”[4] As this report and its annex, jointly authored by the American Civil Liberties Union and Human Rights Watch, demonstrate, the US has a great deal of work ahead to realize the promises of the ICERD.[5] This report therefore includes important recommendations in areas in which our organizations have specialized expertise, for addressing some of the US’s most flagrant violations of the Convention. Additionally, although we recognize the political challenges facing the Biden Administration as it seeks to promote racial equity, we note that its recent declarations on racial justice are disconnected from the government’s longstanding human rights obligations under the ICERD. The attached annex therefore identifies additional, more targeted measures to align President Biden’s commitments with the ICERD. These measures are both fully within the control of the executive branch of the federal government and powerful enough to give life to the President’s stated commitment to put “every branch of the White House and the federal government” to work in the effort to “eliminate systemic racism” in the United States.[6]

President Biden has declared that racism, xenophobia, nativism, and other forms of intolerance are “global problems.”[7] The ICERD is an important part of the solution and to confront these global problems effectively, the US should fulfill its obligations under the treaty.[8] This report and its detailed annex offer an initial roadmap for the US to do exactly that.
    """
    print(summarize(article))
