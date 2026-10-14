"""Build the ISSP 2017 question bank (Social Networks and Social Resources): one row per item v1-v67 with the
source-questionnaire wording (raw/ZA6980_bq.pdf, July 2016), answer options from the .dta value labels, and a
question type. Output: data/issp2017/question_bank.csv. Shared logic: data/issp_common.py.

Q34's printed answer "It does not apply" (v63, v64) shares code 0 with "not asked in this country", so it cannot be
kept as an answer and drops out with the non-answers."""
import sys
from pathlib import Path

HERE = Path(__file__).parent                     # data/issp2017/
sys.path.insert(0, str(HERE.parent))
from issp_common import HIGH_FIRST, write_question_bank  # noqa: E402

KNOW = ("Here is a list of jobs that people you know may have. These people could be family or relatives, close friends "
        "or someone else you know. By \"knowing\" a person, we mean that you know him/her by name and well enough to "
        "contact him/her. If you know several people who have this job, please only tick the box for the person who you "
        "feel closest to. The job could be held by a woman or a man. Do you know a woman or a man who is…")
AGREE = "To what extent do you agree or disagree with the following statement?"
ACTIVITIES = ("Some activities are done with others in organised groups, clubs or associations. In the past 12 months, how "
              "often, if at all, have you taken part in activities…")
TURN_TO = ("This section is about who you would turn to for help in different situations, if you needed it. Please say who "
           "you would turn to first. If there are several people you are equally likely to turn to, please choose the one "
           "you feel closest to. Who would you turn to first to …")
WHERE_TO = ("Please say who or where you would turn to first for help. If there are several choices you are equally likely "
            "to make, please choose the one you would try first. Who or where would you turn to first to …")
FELT = ("The next questions are about how you feel about different aspects of your life. How often in the past 4 weeks "
        "have you felt that…")
TRUST = ("Using the following scale ranging from 0 to 10, where 0 means \"No trust at all\" and 10 means \"Complete trust\", "
         "please indicate how much trust you personally have in…")
CONTACT = "either face-to-face, by phone, internet or any other communication device?"
WEEKS = "During the past 4 weeks how often…"
OUT = ("When you go out to eat or drink with three or more friends or acquaintances who are not family members, how often "
       "do you experience the following situation? How often …")

SOURCE = {
    "v1": f"{KNOW} a bus/lorry driver?",
    "v2": f"{KNOW} a senior executive of a large company?",
    "v3": f"{KNOW} a home or office cleaner?",
    "v4": f"{KNOW} a hairdresser/barber?",
    "v5": f"{KNOW} a human resource manager/personnel manager?",
    "v6": f"{KNOW} a lawyer?",
    "v7": f"{KNOW} a car mechanic?",
    "v8": f"{KNOW} a nurse?",
    "v9": f"{KNOW} a police officer?",
    "v10": f"{KNOW} a school teacher?",
    "v11": f"{AGREE} Differences in income in {{country}} are too large.",
    "v12": f"{AGREE} For a society to be fair, differences in people's standard of living should be small.",
    "v13": f"{AGREE} It is the responsibility of the government to reduce the differences in income between people with high incomes and those with low incomes.",
    "v14": f"{AGREE} The social benefits in {{country}} make people lazy.",
    "v15": "People have different opinions on who should provide services in {country}. Who do you think should primarily provide health care for the sick?",
    "v16": "People have different opinions on who should provide services in {country}. Who do you think should primarily provide care for older people?",
    "v17": f"{ACTIVITIES} … of groups or associations for leisure, sports or culture?",
    "v18": f"{ACTIVITIES} … of political parties, political groups or political associations?",
    "v19": f"{ACTIVITIES} … of charitable or religious organisations that do voluntary work?",
    "v20": f"{AGREE} People like me don't have any say about what the government does.",
    "v21": f"{TURN_TO} … help you with a household or a garden job that you can't do yourself?",
    "v22": f"{TURN_TO} … help you around your home if you were sick and had to stay in bed for a few days?",
    "v23": f"{TURN_TO} … be there for you if you felt a bit down or depressed and wanted to talk about it?",
    "v24": f"{TURN_TO} … give you advice about family problems?",
    "v25": f"{TURN_TO} … enjoy a pleasant social occasion with?",
    "v26": f"{WHERE_TO} … help you if you needed to borrow a large sum of money?",
    "v27": f"{WHERE_TO} … help you if you needed to find a job?",
    "v28": f"{WHERE_TO} … help you with administrative problems or official paperwork?",
    "v29": f"{WHERE_TO} … help you if you needed to find a place to live?",
    "v30": f"{WHERE_TO} … look after you if you were seriously ill?",
    "v31": f"{FELT} … you lack companionship?",
    "v32": f"{FELT} … you are isolated from others?",
    "v33": f"{FELT} … you are left out?",
    "v34": "How often do you think that people would try to take advantage of you if they got the chance, and how often would they try to be fair?",
    "v35": "Generally speaking, would you say that people can be trusted or that you can't be too careful in dealing with people?",
    "v36": f"{TRUST} ... {{country}}'s courts",
    "v37": f"{TRUST} … major private companies in {{country}}",
    "v38": f"{AGREE} Adult children have a duty to look after their elderly parents.",
    "v39": f"{AGREE} You should take care of yourself and your family first, before helping other people.",
    "v40": f"{AGREE} People who are better off should help friends who are less well off.",
    "v41": "In general, do your family members put pressure on you about the way you live or organise your personal life?",
    "v42": "Do you feel that your family, relatives and/or friends make too many demands on you?",
    "v43": "Thinking about the important people in your life, such as your spouse or partner, your family members, or close friends, how often in the past 4 weeks did any of these people act angry or upset with you?",
    "v44": "The following questions are about your social activities. How often do you go out to eat or drink with three or more friends or acquaintances who are not family members?",
    "v45": "When you go out to eat or drink with three or more friends or acquaintances who are not family members, how often do you make new friends or acquaintances?",
    "v46": "Please indicate about how many people do you have contact with on a typical weekday irrespective of whether you know them or not. Include anyone you chat with, talk to, or text, either face-to-face, by phone, internet or any other communication device.",
    "v47": "About how many of the people you have contact with on a typical weekday do you see face-to-face?",
    "v48": f"Please think about the parent you have contact with most frequently: How often do you have contact with that parent, {CONTACT}",
    "v49": f"Think about the brother or sister you have contact with most frequently: How often do you have contact with that brother or sister, {CONTACT}",
    "v50": f"Think about your adult child you have contact with most frequently: How often do you have contact with this child aged at least 18, {CONTACT}",
    "v51": f"Now, think about the other family member you have contact with most frequently, aside of your spouse or partner, parents, siblings or adult children. How often do you have contact with that family member, {CONTACT}",
    "v52": f"And, think about the close friend you have contact with most frequently: How often do you have contact with that close friend, {CONTACT}",
    "v53": "Think now of your contact with all of your family members and close friends. How much of it is through text messages, mobile phones, or other communication devices that use the internet?",
    "v54": "In general, would you say your health is…",
    "v55": f"{WEEKS} … have you felt unhappy and depressed?",
    "v56": f"{WEEKS} … have you felt difficulties were piling up so high that you could not overcome them?",
    "v57": "To what extent is the following statement true or untrue for you? It is easy for me to accomplish my goals.",
    "v58": "All things considered, how satisfied are you with your life as a whole nowadays?",
    "v59": "Thinking of your household's total income, including all the sources of income of all the members who contribute to it, how difficult or easy is it currently for your household to make ends meet?",
    "v60": "How many languages do you speak well enough to hold a conversation in, including the language(s) you speak at home?",
    "v61": f"{AGREE} It is all right to develop friendships with people just because they can be of use to you.",
    "v62": f"{AGREE} When another person does a favour for you, you should feel obligated to pay that person back.",
    "v63": f"{OUT} … does one person dominate the conversation at these occasions?",
    "v64": f"{OUT} … is consideration given to who sits near whom?",
    "v65": f"Please think about the person from your home town or region of origin, other than your family members, you have contact with most frequently. How often do you have contact with that person, {CONTACT}",
    "v66": f"Now think about the person from your religious community, other than your family members, you have contact with most frequently. How often do you have contact with that person, {CONTACT}",
    "v67": "What is the highest level of education that your spouse or partner attained?",
}

CHOICE = {*(f"v{i}" for i in range(1, 11)), "v15", "v16", *(f"v{i}" for i in range(21, 31)),
          "v48", "v49", "v50", "v51", "v52", "v53", "v66"}
EXCLUDE = {"v67": "background variable (spouse's education)"}
LOW_FIRST = {"v59"}   # "Very difficult" ... "Very easy" already runs low -> high

if __name__ == "__main__":
    write_question_bank(HERE, HERE / "raw" / "ZA6980_v2-0-0.dta", HERE / "raw" / "ZA6980_v2-0-0_missing.txt",
                        SOURCE, CHOICE, (), EXCLUDE,
                        HIGH_FIRST + ("Once a week or more", "Daily", "All or almost all", "Completely true",
                                      "Excellent", "People can almost always"), LOW_FIRST)
