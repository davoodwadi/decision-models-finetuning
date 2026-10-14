"""Build the ISSP 2022 question bank (Family and Changing Gender Roles V): one row per item v1-v72 with the
source-questionnaire wording (raw/ZA10000_bq.pdf, August 2021), answer options from the .dta value labels, and a
question type. Output: data/issp2022/question_bank.csv. Shared logic: data/issp_common.py."""
import sys
from pathlib import Path

HERE = Path(__file__).parent                     # data/issp2022/
sys.path.insert(0, str(HERE.parent))
from issp_common import HIGH_FIRST, write_question_bank  # noqa: E402

AGREE = "To what extent do you agree or disagree with the following statement?"
WOMEN = f"To begin, we have some questions about women. {AGREE}"
FAMILIES = f"Children grow up in different kinds of families. {AGREE}"
WORK = "Do you think that women should work full-time, part-time or not at all under the following circumstances?"
PARENT = ("Please consider a family with a father and a mother raising a child who is five years old. In your opinion, "
          "which parent should do the following? Who should …")
LAYOUT = ("Consider a family with a child under school age. What, in your opinion, is the best way for them to organise "
          "their family and work life?")
CHORES = "In your household who does the following things...?"
TIRED = "How often has each of the following happened to you during the past three months?"
SCALE_7 = "Please choose a number on a scale of 1 to 7 where 1 is \"Yes, they definitely should\" and 7 is \"No, they definitely should not\"."
LEADER = ("People may have different opinions about whether women or men are best suited to serve in certain leadership "
          "positions. Please consider the following position and tick the option that comes closest to your own opinion.")
CARE = "Do you provide help or care to the following family member(s), who may or may not be living in your household, on a daily basis?"
ACCEPT = "Think about the time when you were planning to work for pay. How accepting were your parents regarding you working for pay?"

SOURCE = {
    "v1": f"{WOMEN} A working mother can establish just as warm and secure a relationship with her children as a mother who does not work.",
    "v2": f"{WOMEN} A pre-school child is likely to suffer if his or her mother works.",
    "v3": f"{WOMEN} All in all, family life suffers when the woman has a full-time job.",
    "v4": f"{WOMEN} A job is all right, but what most women really want is a home and children.",
    "v5": f"{WOMEN} Being a housewife is just as fulfilling as working for pay.",
    "v6": f"{WOMEN} A man's job is to earn money; a woman's job is to look after the home and family.",
    "v7": "People have different opinions about how much responsibility women and men should have for the home and family. Which of the following statements comes closest to your opinion?",
    "v8": "People have different opinions about how well mothers and fathers are suited to look after their children. Which of the following statements comes closest to your opinion?",
    "v9": f"{WORK} When there is a child under school age.",
    "v10": f"{WORK} After the youngest child starts school.",
    "v11": f"{AGREE} Having children interferes too much with the freedom of parents.",
    "v12": f"{AGREE} Having children increases people's social standing in society.",
    "v13": f"{AGREE} People who want children ought to get married.",
    "v14": f"{AGREE} It is all right for a couple to live together without intending to get married.",
    "v15": f"{FAMILIES} A single mother can bring up her child as well as a married couple.",
    "v16": f"{FAMILIES} A single father can bring up his child as well as a married couple.",
    "v17": f"{FAMILIES} A same sex female couple can bring up a child as well as a male-female couple.",
    "v18": f"{FAMILIES} A same sex male couple can bring up a child as well as a male-female couple.",
    "v19": "All in all, what do you think is the ideal number of children for a family to have?",
    "v20": f"{PARENT} … provide for the family financially?",
    "v21": f"{PARENT} … take care of the child on a daily basis?",
    "v22": f"{PARENT} … play with the child, and take part in his/her leisure activities?",
    "v23": f"{PARENT} … teach the child how to behave?",
    "v24": f"{PARENT} … take time to listen to and advise the child if he/she has problems?",
    "v25": f"{PARENT} … try to be a role model for the child?",
    "v26": "Consider a couple who both work full-time and now have a new born child. One of them stops working for some time to care for their child. Do you think there should be paid leave available and, if so, for how long?",
    "v27": "Consider a couple who both work full-time and now have a new born child. One of them stops working for some time to care for their child. If both are in a similar work situation and are eligible for paid leave, how should this paid leave period be divided between the mother and the father?",
    "v28": LAYOUT,
    "v29": f"{LAYOUT} Which of these options would be the least desirable?",
    "v30": "People have different views on childcare for children under school age. Who do you think should primarily provide childcare?",
    "v31": "Who do you think should primarily cover the costs of childcare for children under school age?",
    "v32": "Thinking about elderly people who need some help in their everyday lives, such as help with grocery shopping, cleaning the house, doing the laundry etc. Who do you think should primarily provide this help?",
    "v33": "Thinking about elderly people who need some help in their everyday lives, such as help with grocery shopping, cleaning the house, doing the laundry etc. Who do you think should primarily cover the costs of this help to these elderly people?",
    "v34": "On average, how many hours a week do you personally spend on household work, not including childcare and leisure time activities?",
    "v35": "On average, how many hours a week do you spend looking after family members (e.g. children, elderly, ill or disabled family members)?",
    "v36": "On average, how many hours a week does your spouse/partner spend on household work, not including childcare and leisure time activities?",
    "v37": "On average, how many hours a week does your spouse/partner spend looking after family members (e.g. children, elderly, ill or disabled family members)?",
    "v38": "How do you and your spouse/partner organise the income that one or both of you receive? Please choose the option that comes closest.",
    "v39": f"{CHORES} Does the laundry",
    "v40": f"{CHORES} Plans and organizes social and family activities",
    "v41": f"{CHORES} Cares for sick family members",
    "v42": f"{CHORES} Shops for groceries",
    "v43": f"{CHORES} Does the household cleaning",
    "v44": f"{CHORES} Prepares the meals",
    "v45": "Which of the following best applies to the sharing of household work between you and your spouse/partner?",
    "v46": f"{TIRED} I have come home from work too tired to do the chores which need to be done.",
    "v47": f"{TIRED} It has been difficult for me to fulfil my family responsibilities because of the amount of time I spent on my job.",
    "v48": f"{TIRED} I have arrived at work too tired to function well because of the household work I had done.",
    "v49": f"{TIRED} I have found it difficult to concentrate at work because of my family responsibilities.",
    "v50": "People may have different perceptions about who they consider to be close relatives. Imagine a person living with a partner and their children. Which of the following options comes closest to who you would consider to be his/her close relatives?",
    "v51": f"People have different opinions about responsibilities towards relatives. Consider a couple living with their two children who has enough space in their home for someone to come and stay. A sibling of one of the couple needs a place to stay for a few months. In your opinion, do you think they should, or should not, offer him/her to stay for free in their home? {SCALE_7}",
    "v52": f"People have different opinions about responsibilities towards relatives. Consider another couple living with their two children who has enough space in their home for someone to come and stay. The father of one of them recently became a widower. Although he is in good health, he is feeling lonely. In your opinion, do you think they should, or should not, ask the father to come and live with them permanently? {SCALE_7}",
    "v53": f"People have different opinions about responsibilities towards relatives. Consider an elderly couple. One of their grandchildren is in need of temporary financial help because the company he/she worked for closed down recently. Do you think they should, or should not, try their best to help their grandchild financially? {SCALE_7}",
    "v54": "People have different opinions about who you can rely on and who should be important to you. While some people think that you can rely more on close relatives, others think that you can rely more on close friends. Which view is closer to your opinion? Please choose a number on a scale of 1 to 7 where 1 is \"You can definitely rely more on close relatives\", 4 is \"You can rely equally on both\" and 7 is \"You can definitely rely more on close friends\" or \"You cannot rely on either of them\".",
    "v55": "While some people think that close relatives should be the most important people in one's life, others think that it should be close friends who are the most important. Which view is closer to your opinion? Please choose a number on a scale of 1 to 7 where 1 is \"Close relatives should definitely be the most important\", 4 is \"Both should be equally important\" and 7 is \"Close friends should definitely be the most important\".",
    "v56": "If you were to consider your life in general, how happy or unhappy would you say you are, on the whole?",
    "v57": f"{LEADER} Cabinet minister in the national government",
    "v58": f"{LEADER} Head of a university",
    "v59": f"{LEADER} Senior executive of a large company",
    "v60": "Thinking about your household's total income, including all the sources of income of all the members who contribute to it, how difficult or easy is it currently for your household to make ends meet?",
    "v61": "Compared to 12 months ago, how do you think the financial situation of your household has changed? Would you say the current situation is …",
    "v62": "Education of current spouse/partner.",
    "v63": "How many children have you ever had? Please include any biological, adoptive, foster or step-children.",
    "v64": f"{CARE} Child(ren) below school age",
    "v65": f"{CARE} Child(ren) of school age, up to 18 years",
    "v66": f"{CARE} Elderly person(s)",
    "v67": f"{CARE} Long term sick/disabled person(s)",
    "v68": f"{CARE} Other person(s)",
    "v69": "By \"parents\" we mean biological parents, step-parents, foster parents or adoptive parents. By \"upbringing\" we mean being involved hands on in the day-to-day raising of a child, for example, giving moral values, emotional support, involvement in day-to-day activities. Who was more involved in your upbringing when you were a child – your mother, your father or both equally?",
    "v70": f"{ACCEPT} My mother was …",
    "v71": f"{ACCEPT} My father was …",
    "v72": "By \"upbringing\" we mean being involved hands on in the day to day raising of your child(ren), for example, giving the child(ren) moral values, emotional support, involvement in day-to-day activities. Now thinking about your children. Who has been more involved in their upbringing, yourself or your spouse/partner?",
}

CHOICE = {"v46", "v47", "v48", "v49", "v28", "v29", "v30", "v31", "v32", "v33", "v38", "v39", "v40", "v41", "v42", "v43", "v44", "v54", "v69", "v72"}
NOUL = {"v64", "v65", "v66", "v67", "v68"}
EXCLUDE = {"v19": "open count (ideal number of children)", "v26": "open count (months of paid leave)",
           "v34": "open count (hours)", "v35": "open count (hours)", "v36": "open count (hours)",
           "v37": "open count (hours)", "v62": "background variable (spouse's education)",
           "v63": "open count (number of children)"}
LOW_FIRST = {"v60"}   # "Very difficult" ... "Very easy" already runs low -> high
# printed answers the data codes as non-answers: kept as answers (the items become `choice`)
EXTRA_ANSWERS = {v: {-4: "Doesn't apply/no job"} for v in ["v46", "v47", "v48", "v49"]}

if __name__ == "__main__":
    write_question_bank(HERE, HERE / "raw" / "ZA10000_v2-0-0.dta", HERE / "raw" / "ZA10000_v2-0-0_missing.txt",
                        SOURCE, CHOICE, NOUL, EXCLUDE, HIGH_FIRST + ("Several times a week",), LOW_FIRST, EXTRA_ANSWERS)
