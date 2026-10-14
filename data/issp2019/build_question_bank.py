"""Build the ISSP 2019 question bank (Social Inequality V): one row per item v1-v70 with the source-questionnaire
wording (raw/ZA7600_bq.pdf, July 2018), answer options from the .dta value labels, and a question type.
Output: data/issp2019/question_bank.csv. Shared logic: data/issp_common.py.

"When you were [14-15-16] years old" is written "when you were about 15 years old" (the age the questionnaire
recommends)."""
import sys
from pathlib import Path

HERE = Path(__file__).parent                     # data/issp2019/
sys.path.insert(0, str(HERE.parent))
from issp_common import HIGH_FIRST, write_question_bank  # noqa: E402

AHEAD = ("To begin we have some questions about opportunities for getting ahead. Please tick one box to show how "
         "important you think the following is for getting ahead in life…")
AGREE = "To what extent do you agree or disagree with the following statement?"
INTERNATIONAL = f"Turning to international differences. {AGREE}"
CONFLICT = ("In all countries, there are differences or even conflicts between different social groups. In your opinion, "
            "in {country} how much conflict is there between…")
TOPBOT = ("In our society there are groups which tend to be towards the top and groups which tend to be towards the "
          "bottom. Below is a scale which runs from top to bottom.")
PAY = "In deciding how much people ought to earn, how important should the following be, in your opinion…"
DIAGRAMS = ("These five diagrams show different types of society. Type A: a small elite at the top, very few people in "
            "the middle and the great mass of people at the bottom. Type B: a society like a pyramid with a small elite "
            "at the top, more people in the middle, and most at the bottom. Type C: a pyramid except that just a few "
            "people are at the bottom. Type D: a society with most people in the middle. Type E: many people near the "
            "top, and only a few near the bottom.")
CONTACT = ("This might be in the street, on public transport, in shops, in your neighbourhood, or at your workplace.")
ACTUAL = ("We would like to know what you think people in these jobs actually earn. Please write in how much you think "
          "they usually earn.")
SHOULD = "What do you think people in these jobs ought to be paid? Please write in how much they should earn."
JOB_TYPES = "Here is a list of different types of jobs."

SOURCE = {
    "v1": f"{AHEAD} …how important is coming from a wealthy family?",
    "v2": f"{AHEAD} …how important is having well-educated parents?",
    "v3": f"{AHEAD} …how important is having a good education yourself?",
    "v4": f"{AHEAD} …how important is hard work?",
    "v5": f"{AHEAD} …how important is knowing the right people?",
    "v6": f"{AHEAD} …how important is having political connections?",
    "v7": f"{AHEAD} …how important is giving bribes?",
    "v8": f"{AHEAD} …how important is a person's race?",
    "v9": f"{AHEAD} …how important is a person's religion?",
    "v10": f"{AHEAD} …how important is being born a man or a woman?",
    "v11": f"{ACTUAL} About how much do you think a doctor in general practice earns?",
    "v12": f"{ACTUAL} How much do you think a chairman of a large national corporation earns?",
    "v13": f"{ACTUAL} How much do you think a shop assistant earns?",
    "v14": f"{ACTUAL} How much do you think an unskilled worker in a factory earns?",
    "v15": f"{ACTUAL} How much do you think a cabinet minister in the national government earns?",
    "v16": f"{SHOULD} About how much do you think a doctor in general practice should earn?",
    "v17": f"{SHOULD} How much do you think a chairman of a large national corporation should earn?",
    "v18": f"{SHOULD} How much do you think a shop assistant should earn?",
    "v19": f"{SHOULD} How much do you think an unskilled worker in a factory should earn?",
    "v20": f"{SHOULD} How much do you think a cabinet minister in the national government should earn?",
    "v21": f"{AGREE} Differences in income in {{country}} are too large.",
    "v22": f"{AGREE} It is the responsibility of the government to reduce the differences in income between people with high incomes and those with low incomes.",
    "v23": f"{AGREE} The government should provide a decent standard of living for the unemployed.",
    "v24": f"{AGREE} It is the responsibility of private companies to reduce the differences in pay between their employees with high pay and those with low pay.",
    "v25": "Looking at the list below, who do you think should have the greatest responsibility for reducing differences in income between people with high incomes and people with low incomes?",
    "v26": "To what extent do you agree or disagree with the following statement: Most politicians in {country} do not care about reducing the differences in income between people with high incomes and people with low incomes.",
    "v27": "How successful do you think the government in {country} is nowadays in reducing the differences in income between people with high incomes and people with low incomes?",
    "v28": "Do you think people with high incomes should pay a larger share of their income in taxes than those with low incomes, the same share, or a smaller share?",
    "v29": "Generally, how would you describe taxes in {country} today for those with high incomes? Taxes are…",
    "v30": "Is it just or unjust – right or wrong – that people with higher incomes can buy better health care than people with lower incomes?",
    "v31": "Is it just or unjust – right or wrong – that people with higher incomes can buy better education for their children than people with lower incomes?",
    "v32": "Some people feel angry about differences in wealth between the rich and the poor, while others do not. How do you feel when you think about differences in wealth between the rich and the poor in {country}? Please place yourself on a scale of 0 to 10, where 0 means not angry at all and 10 means extremely angry.",
    "v33": f"{INTERNATIONAL} Present economic differences between rich and poor countries are too large.",
    "v34": f"{INTERNATIONAL} People in wealthy countries should make an additional tax contribution to help people in poor countries.",
    "v35": f"{INTERNATIONAL} People from poor countries should be allowed to work in wealthy countries.",
    "v36": f"{CONFLICT} … poor people and rich people?",
    "v37": f"{CONFLICT} … the working class and the middle class?",
    "v38": f"{CONFLICT} … management and workers?",
    "v39": f"{CONFLICT} … young people and older people?",
    "v40": f"{CONFLICT} … people born in {{country}} and people from other countries who have come to live in {{country}}?",
    "v41": f"{TOPBOT} Where would you put yourself now on this scale?",
    "v42": f"{TOPBOT} If you think about the family that you grew up in, where did they fit in then?",
    "v43": f"{TOPBOT} Thinking ahead 10 years from now, where do you think you will be on a scale of 1 to 10, where 10 is the top and 1 the bottom?",
    "v44": f"{PAY} How much responsibility goes with the job – how important do you think that ought to be in deciding pay?",
    "v45": f"{PAY} … the number of years spent in education and training?",
    "v46": f"{PAY} … whether the person has children to support – how important should that be in deciding pay?",
    "v47": f"{PAY} … how well he or she does the job – how important should that be in deciding pay?",
    "v48": f"{DIAGRAMS} First, what type of society is {{country}} today – which diagram comes closest?",
    "v49": f"{DIAGRAMS} What do you think {{country}} ought to be like – which would you prefer?",
    "v50": "How fair or unfair do you think the income distribution is in {country}?",
    "v51": f"How often do you have any contact with people who are a lot poorer than you when you are out and about? {CONTACT}",
    "v52": f"How often do you have any contact with people who are a lot richer than you when you are out and about? {CONTACT}",
    "v53": "Thinking of your household's total income, including all the sources of income of all the members who contribute to it, how difficult or easy is it currently for your household to make ends meet?",
    "v54": "And during the next 12 months, how difficult or easy do you think it will it be for your household to make ends meet?",
    "v55": "How often do you or other members of your household skip a meal because there is not enough money for food?",
    "v56": "Generally speaking, would you say that people can be trusted or that you can't be too careful in dealing with people?",
    "v57": "When you were about 15 years old, was your father an employee, self-employed, or working for his own family's business? If your father did not have a paid job at the time, please give information about his last job before that time.",
    "v58": "When you were about 15 years old, was your mother an employee, self-employed, or working for her own family's business? If your mother did not have a paid job at the time, please give information about her last job before that time.",
    "v59": "When you were about 15 years old, what kind of work did your father do; what was his main occupation?",
    "v60": "When you were about 15 years old, what kind of work did your mother do; what was her main occupation?",
    "v61": "Most people see themselves as belonging to a particular class. Please tell me which social class you would say you belong to?",
    "v62": f"{AGREE} Workers would not bother to get skills and qualifications unless they were paid extra for having them.",
    "v63": f"{AGREE} Large differences in income are necessary for {{country}}'s prosperity.",
    "v64": f"{AGREE} Inequality continues because it benefits the rich and powerful.",
    "v65": f"{AGREE} Inequality continues to exist because ordinary people don't join together to get rid of it.",
    "v66": "To what extent do you think that differences in wealth between rich and poor countries are fair or unfair?",
    "v67": f"{JOB_TYPES} Which type of job did your father have when you were about 15 years old? If your father did not have a paid job at the time, please give information about his last job before that time.",
    "v68": f"{JOB_TYPES} Which type of job did your mother have when you were about 15 years old? If your mother did not have a paid job at the time, please give information about her last job before that time.",
    "v69": f"{JOB_TYPES} Which type of job do you have now in your current job? If you are not working now, please tell us about your last job.",
    "v70": f"{JOB_TYPES} And which type of job does your spouse/partner have now in her/his current job? If she/he is not working now, please tell us about her/his last job.",
}

CHOICE = {"v25", "v48", "v49", "v57", "v58", "v67", "v68", "v69", "v70"}
EXCLUDE = {**{f"v{i}": "open amount (earnings)" for i in range(11, 21)},
           "v59": "open occupation (coded to ISCO, hundreds of codes)", "v60": "open occupation (coded to ISCO, hundreds of codes)"}
LOW_FIRST = {"v53", "v54"}   # "Very difficult" ... "Very easy" already runs low -> high
# Q20a/b code 6 "Father/Mother unknown" is "Don't know/ Don't remember" in Sweden (value label note): a non-answer there
COUNTRY_NOT_ANSWERS = {"v57": {"SE": [6]}, "v58": {"SE": [6]}}

if __name__ == "__main__":
    write_question_bank(HERE, HERE / "raw" / "ZA7600_v3-0-0.dta", HERE / "raw" / "ZA7600_v3-0-0_missing.txt",
                        SOURCE, CHOICE, (), EXCLUDE,
                        HIGH_FIRST + ("Essential", "Quite successful", "People can almost always"), LOW_FIRST,
                        country_not_answers=COUNTRY_NOT_ANSWERS)
