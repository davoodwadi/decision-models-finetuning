"""Build the ISSP 2021 question bank (Health and Health Care II): one row per item v1-v70 with the
source-questionnaire wording (raw/ZA8000_bq.pdf, November 2020), answer options from the .dta value labels, and a
question type. Output: data/issp2021/question_bank.csv. Shared logic: data/issp_common.py.

The questionnaire lets each country name its own alternative and conventional medicine ("[Alternative/traditional/
folk] medicine"); the wording here uses "alternative/traditional/folk" and "mainstream/Western conventional"."""
import sys
from pathlib import Path

HERE = Path(__file__).parent                     # data/issp2021/
sys.path.insert(0, str(HERE.parent))
from issp_common import HIGH_FIRST, write_question_bank  # noqa: E402

AGREE = "How much do you agree or disagree with the following statement?"
ACCESS = f"{AGREE} People should have access to publicly funded health care even if they…"
EASIER = "In {country}, do you think it is easier or harder to get access to health care …"
CAUSES = f"Severe health problems may have many causes. {AGREE} People suffer from severe health problems…"
DOCTORS = "How much do you agree or disagree with the following statement about doctors in general in {country}?"
TOPICS = "During the past 12 months, how often, if at all, have you used the internet to look for information on the following topic?"
INTERNET_INFO = f"To what extent do you agree or disagree with the following statement? During the past 12 months, information on the internet…"
EXTENT = "To what extent do you agree or disagree with the following statement?"
VACCINES = "How much do you agree or disagree with the following statement about vaccinations?"
WEEKS = "During the past 4 weeks how often…"
VISITS = "During the past 12 months, how often did you visit or were visited by..."
UNTREATED = "During the past 12 months did it ever happen that you did not get the medical treatment you needed because…"
TREATMENT = "How satisfied or dissatisfied were you with the treatment you received…"
OFTEN = "How often do you…"
EPIDEMIC = "Do you think {country}'s government should or should not have the right to do the following at times of severe epidemics?"
COVID = "Did the way the Covid-19 pandemic was handled in {country} increase or decrease your confidence in…"
STIGMA = f"There are different opinions about people that have certain health conditions. {AGREE}"

SOURCE = {
    "v1": "If you were to consider your life in general these days, how happy or unhappy would you say you are, on the whole?",
    "v2": "In general, how much confidence do you have in the health care system in {country}?",
    "v3": "Is it fair or unfair that people with higher incomes can afford better health care than people with lower incomes?",
    "v4": f"{AGREE} People use health care services more than necessary.",
    "v5": f"{AGREE} The government should provide only limited health care services.",
    "v6": f"{AGREE} In general, the health care system in {{country}} is inefficient.",
    "v7": "How willing would you be to pay higher taxes to improve the level of health care for all people in {country}?",
    "v8": f"{ACCESS} do not hold citizenship of {{country}}.",
    "v9": f"{ACCESS} behave in ways that damage their health.",
    "v10": f"{EASIER} for rich people than for poor people",
    "v11": f"{EASIER} for old people than for young people",
    "v12": f"{EASIER} for women than for men",
    "v13": f"{EASIER} for citizens of {{country}} than for people who do not hold {{country}}'s citizenship",
    "v14": f"{CAUSES} because they behaved in ways that damaged their health.",
    "v15": f"{CAUSES} because of the environment they are exposed to at work or where they live.",
    "v16": f"{CAUSES} because of their genes.",
    "v17": f"{CAUSES} because they are poor.",
    "v18": "To what extent do you agree or disagree with the following statement: Alternative/traditional/folk medicine provides better solutions for health problems than mainstream/Western conventional medicine?",
    "v19": f"{DOCTORS} All things considered, doctors can be trusted.",
    "v20": f"{DOCTORS} The medical skills of doctors are not as good as they should be.",
    "v21": f"{DOCTORS} Doctors care more about their earnings than about their patients.",
    "v22": "During the past 12 months, how often, if at all, did you use the internet on any device (such as computers, tablets and smartphones) to look for health or medical information for yourself or someone else?",
    "v23": f"{TOPICS} Information on healthy lifestyle?",
    "v24": f"{TOPICS} Information related to anxiety, stress, or similar problems?",
    "v25": f"{TOPICS} Information on vaccinations?",
    "v26": f"{INTERNET_INFO} affected my health behaviour in a positive way.",
    "v27": f"{INTERNET_INFO} helped me understand what a doctor tried to explain to me.",
    "v28": f"{EXTENT} The internet is useful to help people decide if their symptoms are serious enough to go to the doctor.",
    "v29": f"{EXTENT} The internet is useful to check that the doctor is giving people appropriate advice.",
    "v30": f"{EXTENT} It is not easy to distinguish between reliable and unreliable health information on the internet.",
    "v31": f"{VACCINES} Overall, vaccinations do more harm than good.",
    "v32": f"{VACCINES} It is better to develop immunity by getting ill than having a vaccination.",
    "v33": f"{WEEKS} have you had difficulties with work or household activities because of health problems?",
    "v34": f"{WEEKS} have you had bodily aches or pains?",
    "v35": f"{WEEKS} have you felt unhappy and depressed?",
    "v36": f"{WEEKS} have you lost confidence in yourself?",
    "v37": f"{WEEKS} have you felt you could not overcome your problems?",
    "v38": f"{VISITS} a doctor?",
    "v39": f"{VISITS} an alternative/traditional/folk health care practitioner?",
    "v40": f"{UNTREATED} you could not pay for it?",
    "v41": f"{UNTREATED} you could not take the time off work or had other commitments?",
    "v42": f"{UNTREATED} the waiting list was too long?",
    "v43": "How likely is it that if you become seriously ill, you would get or not get the best treatment available in {country}?",
    "v44": "In general, how satisfied or dissatisfied are you with the health care system in {country}?",
    "v45": f"{TREATMENT} when you last visited a doctor?",
    "v46": f"{TREATMENT} when you last visited an alternative/traditional/folk health care practitioner?",
    "v47": "Do you smoke cigarettes, and if so about how many cigarettes a day?",
    "v48": f"{OFTEN} drink 4 or more alcoholic drinks on the same day?",
    "v49": f"{OFTEN} do physical activity for at least 20 minutes that makes you sweat or breathe more heavily than usual?",
    "v50": f"{OFTEN} eat fresh fruit or vegetables?",
    "v51": "In general, would you say your health is …",
    "v52": "Do you have a long-standing illness, a chronic condition, or a disability?",
    "v53": "What is your height?",
    "v54": "What is your weight?",
    "v55": f"{EPIDEMIC} Shut down businesses and places of employment.",
    "v56": f"{EPIDEMIC} Demand that people stay at home.",
    "v57": f"{EPIDEMIC} Use digital (mobile phone) surveillance to track infected people.",
    "v58": f"{EPIDEMIC} Require people to wear face masks.",
    "v59": f"{EPIDEMIC} Ban public gatherings.",
    "v60": "What kind of health insurance do you have?",
    "v61": f"{EPIDEMIC} Place people known to carry the disease in isolation.",
    "v62": f"{EPIDEMIC} Suspend compulsory education and close schools and kindergartens.",
    "v63": f"{EPIDEMIC} Close borders to other countries.",
    "v64": f"{COVID} the health care system?",
    "v65": f"{COVID} the government?",
    "v66": "Considering your work activity before the Covid-19 pandemic and at present, which of the following statements best describes your employment situation…",
    "v67": "Thinking about the income of your household before the Covid-19 pandemic compared with now, would you say it has increased, decreased or stayed about the same?",
    "v68": "Thinking about how often you met with your extended family and friends in person before the Covid-19 pandemic, would you say you now meet them in person more often, less often, or about the same?",
    "v69": f"{STIGMA} Most people become very overweight because they are lazy.",
    "v70": f"{STIGMA} Most people who tested positive for Covid-19 contracted the virus because they were careless.",
}

# printed answers the data codes as non-answers: kept as answers (the items become `choice`)
EXTRA_ANSWERS = {"v22": {-4: "I do not have access to the internet"},
                 **{v: {-4: "Did not need medical treatment"} for v in ["v40", "v41", "v42"]},
                 **{v: {-4: "Does not apply"} for v in ["v45", "v46"]}}
CHOICE = {"v22", "v40", "v41", "v42", "v45", "v46", "v60", "v66"}
NOUL = {"v52"}
EXCLUDE = {"v53": "open measurement (height)", "v54": "open measurement (weight)"}
LOW_FIRST = {"v68"}   # "Much less often" ... "Much more often" already runs low -> high

if __name__ == "__main__":
    write_question_bank(HERE, HERE / "raw" / "ZA8000_v2-0-0.dta", HERE / "raw" / "ZA8000_v2-0-0_missing.txt",
                        SOURCE, CHOICE, NOUL, EXCLUDE, HIGH_FIRST + ("Complete ", "Excellent", "Increased", "It's certain I would get"),
                        LOW_FIRST, EXTRA_ANSWERS)
