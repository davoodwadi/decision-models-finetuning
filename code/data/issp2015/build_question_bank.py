"""Build the ISSP 2015 question bank (Work Orientations IV): one row per item with the source-questionnaire wording
(raw/ZA6770_bq.pdf, July 2014), answer options from the .dta value labels, and a question type.
Output: data/issp2015/question_bank.csv. Shared logic: data/issp_common.py.

Most of this questionnaire is split: Q11-Q31 were asked only of people working for pay, Q32-Q44 only of people not
working for pay (see ASKED_OF in build_dataset.py). Q7 is stored in the data as "V16" (capital V)."""
import sys
from pathlib import Path

HERE = Path(__file__).parent                     # data/issp2015/
sys.path.insert(0, str(HERE.parent))
from issp_common import HIGH_FIRST, write_question_bank  # noqa: E402

WORK = ("Please tick one box for each statement below to show how much you agree or disagree with it, thinking of work "
        "in general.")
IMPORTANT = "Please show how important you personally think the following is in a job. How important is..."
AGREE = "To what extent do you agree or disagree with the following statement?"
MY_JOB = "For the following statement about your (main) job, please show how much you agree or disagree."
HOW_OFTEN = "Now some more questions about your working conditions. Please show how often the following applies to your work. How often..."
FEEL = "How often do you feel that …"
RELATIONS = "In general, how would you describe relations at your workplace…"
FIRM = AGREE
OCCUPATION = ("Now think of the type of work you do, irrespective of the firm or organization you work for. "
              f"{AGREE}")
AVOID = f"{AGREE} In order to avoid unemployment I would be willing …"
GET_JOB = f"{AGREE} In order to get a job I would be willing …"
SEEKING = "Thinking about the past 12 months, have you done the following in order to find a job?"
CHANGES = "Over the past five years, did you experience the following change in your working life?"
WORRY = "To what extent, if at all, do you worry about the possibility of"

SOURCE = {
    "v1": f"{WORK} A job is just a way of earning money - no more.",
    "v2": f"{WORK} I would enjoy having a paid job even if I did not need the money.",
    "v3": f"{IMPORTANT} … job security",
    "v4": f"{IMPORTANT} … high income",
    "v5": f"{IMPORTANT} … good opportunities for advancement",
    "v6": f"{IMPORTANT} … an interesting job",
    "v7": f"{IMPORTANT} … a job that allows someone to work independently",
    "v8": f"{IMPORTANT} … a job that allows someone to help other people",
    "v9": f"{IMPORTANT} … a job that is useful to society",
    "v10": f"{IMPORTANT} … a job that allows someone to decide their times or days of work",
    "v11": f"{IMPORTANT} … a job that involves personal contact with other people",
    "v12": "Have you ever given up or would you give up good job opportunities for the benefit of your family life?",
    "v13": "Have you ever remained or would you remain in a job that was not satisfying for you for the benefit of your family life?",
    "v14": "Over the past five years, have you been discriminated against with regard to work, for instance when applying for a job, or when being considered for a pay increase or promotion?",
    "v15": "In your opinion, what was the main reason for the discrimination?",
    "V16": "Over the past five years, have you been harassed by your superiors or co-workers at your job, for example, have you experienced any bullying, physical or psychological abuse?",
    "v17": f"{AGREE} Workers need strong trade unions to protect their interests.",
    "v18": f"{AGREE} Strong trade unions are bad for {{country}}'s economy.",
    "v19": "Suppose you could decide on your work situation at present. Which of the following would you prefer?",
    "v20": "Are you currently working for pay?",
    "v21": "Think of the number of hours you work, and the money you earn in your main job, including any regular overtime. If you had only one of these three choices, which of the following would you prefer?",
    "v22": f"{MY_JOB} My job is secure.",
    "v23": f"{MY_JOB} My income is high.",
    "v24": f"{MY_JOB} My opportunities for advancement are high.",
    "v25": f"{MY_JOB} My job is interesting.",
    "v26": f"{MY_JOB} I can work independently.",
    "v27": f"{MY_JOB} In my job I can help other people.",
    "v28": f"{MY_JOB} My job is useful to society.",
    "v29": f"{MY_JOB} In my job, I have personal contact with other people.",
    "v30": f"{HOW_OFTEN} ... do you have to do hard physical work?",
    "v31": f"{HOW_OFTEN} ... do you find your work stressful?",
    "v32": f"{HOW_OFTEN} … do you work at home during your usual working hours?",
    "v33": f"{HOW_OFTEN} … does your job involve working on weekends?",
    "v34": "Which of the following statements best describes how your working hours are decided? (By working hours we mean here the times you start and finish work, and not the total hours you work per week or month.)",
    "v35": "Which of the following statements best describes your usual working schedule in your main job?",
    "v36": "Which of the following statements best describes how your daily work is organized?",
    "v37": "How difficult would it be for you to take an hour or two off during working hours, to take care of personal or family matters?",
    "v38": f"{FEEL} …the demands of your job interfere with your family life?",
    "v39": f"{FEEL} …the demands of your family life interfere with your job?",
    "v40": "How much of your past work experience and/or job skills can you make use of in your present job?",
    "v41": "Over the past 12 months, have you had any training to improve your job skills, either at the workplace or somewhere else?",
    "v42": f"{RELATIONS} ... between management and employees?",
    "v43": f"{RELATIONS} ... between workmates/colleagues?",
    "v44": "How satisfied are you in your (main) job?",
    "v45": f"{FIRM} I am willing to work harder than I have to in order to help the firm or organization I work for succeed.",
    "v46": f"{FIRM} I am proud to be working for my firm or organization.",
    "v47": f"{FIRM} I would turn down another job that offered quite a bit more pay in order to stay with this organization.",
    "v48": f"{OCCUPATION} Given the chance, I would change my present type of work for something different.",
    "v49": f"{OCCUPATION} I am proud of the type of work I do.",
    "v50": "How difficult or easy do you think it would be for you to find a job at least as good as your current one?",
    "v51": "All in all, how likely is it that you will try to find a job with another firm or organization within the next 12 months?",
    "v52": f"{WORRY} losing your job?",
    "v53": f"{AVOID} to accept a job that requires new skills.",
    "v54": f"{AVOID} to accept a position with lower pay.",
    "v55": f"{AVOID} to accept temporary employment.",
    "v56": f"{AVOID} to travel longer to get to work.",
    "v57": f"{AVOID} to move within {{country}}.",
    "v58": f"{AVOID} to move to a different country.",
    "v59": "Over the past 12 months, in addition to your main job, have you done any other work for pay?",
    "v60": "Over the entire 12 months, how much did you earn from your additional job(s) in total compared with your main job? From my additional job(s), I earned …",
    "v61": "Have you ever had a paid job for one year or more?",
    "v62": "When did your last paid job end? (year)",
    "v63": "When did your last paid job end? (month)",
    "v64": "How satisfied were you in your last job?",
    "v65": "What was the main reason that your job ended?",
    "v66": "Would you like to have a paid job, either now or in the future?",
    "v67": "How likely do you think it is that you would find a job?",
    "v68": f"{WORRY} not finding a job?",
    "v69": f"{GET_JOB} to accept a job that requires new skills.",
    "v70": f"{GET_JOB} to accept a position with low pay.",
    "v71": f"{GET_JOB} to accept temporary employment.",
    "v72": f"{GET_JOB} to travel a long time to get to work.",
    "v73": f"{GET_JOB} to move within {{country}}.",
    "v74": f"{GET_JOB} to move to a different country.",
    "v75": f"{SEEKING} Registered at a public employment agency?",
    "v76": f"{SEEKING} Registered at a private employment agency?",
    "v77": f"{SEEKING} Answered advertisements for jobs?",
    "v78": f"{SEEKING} Advertised yourself for a job, for example on the internet or in newspapers?",
    "v79": f"{SEEKING} Applied directly to employers?",
    "v80": f"{SEEKING} Asked relatives, friends, or colleagues to help you find a job?",
    "v81": "Over the past 12 months, have you had any training to improve your job skills?",
    "v82": "Are you currently looking for a job?",
    "v83": "What is your main source of economic support?",
    "v84": f"{WORRY} losing this main economic support?",
    "v85": "What is the age of the youngest child in your household?",
    "v86": "Over the past five years, have you ever worked for pay?",
    "v87": f"{CHANGES} I was unemployed for a period longer than three months.",
    "v88": f"{CHANGES} I changed my employer.",
    "v89": f"{CHANGES} I changed my occupation.",
    "v90": f"{CHANGES} I started my own business/became self-employed.",
    "v91": f"{CHANGES} I took up an additional job.",
    "v92": "How would you generally rate your current financial situation? My current financial situation is…",
    "v93": "Compared to five years ago, how do you think your financial situation has changed? My current financial situation is…",
    "v94": "And what will your financial situation likely be in five years? My financial situation in five years will likely be…",
    "v95": f"{AGREE} It is good for {{country}}'s economy that people aged 60 and over are employed.",
    "v96": f"{AGREE} When people aged 60 and over are employed, they take jobs away from younger people.",
    "v97": "In general, would you say your health is …",
}

# printed answers the data codes as non-answers: kept as answers (the items become `choice`)
EXTRA_ANSWERS = {"v14": {0: "Did not work or did not seek work"},
                 "V16": {0: "Does not apply (I do not have a job/ superior/ co-worker)"}}
CHOICE = {"v12", "v13", "v14", "v15", "V16", "v35", "v65", "v83"}
NOUL = {"v20", "v41", "v61", "v66", "v81", "v82", "v86"}
EXCLUDE = {"v62": "open date (year last job ended)", "v63": "open date (month last job ended)",
           "v85": "open count (age of youngest child)"}

if __name__ == "__main__":
    write_question_bank(HERE, HERE / "raw" / "ZA6770_v2-1-0.dta", HERE / "raw" / "ZA6770_v2-1-0_missing.txt",
                        SOURCE, CHOICE, NOUL, EXCLUDE,
                        HIGH_FIRST + ("I worry a great deal", "Excellent"), (), EXTRA_ANSWERS)
