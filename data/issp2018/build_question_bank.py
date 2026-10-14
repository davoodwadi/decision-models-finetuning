"""Build the ISSP 2018 question bank (Religion IV): one row per item v1-v81 with the source-questionnaire wording
(raw/ZA7570_bq.pdf, June 2017), answer options from the .dta value labels, and a question type.
Output: data/issp2018/question_bank.csv. Shared logic: data/issp_common.py.

The country-specific religion items (v34, v36, v38: each country's own list of religions) are not v-numbered items in
the data's common part; the harmonised versions (v35, v37, v39: religious groups) are used instead. Where the
questionnaire asks for country-specific examples ("[COUNTRY-SPECIFIC LIST icon, retablo, mezuzah, menorah, or
crucifix]"), the source questionnaire's examples are used."""
import sys
from pathlib import Path

HERE = Path(__file__).parent                     # data/issp2018/
sys.path.insert(0, str(HERE.parent))
from issp_common import HIGH_FIRST, write_question_bank  # noqa: E402

CONFIDENCE = "How much confidence do you have in ..."
AGREE = "Do you agree or disagree with the following statement?"
EXTREMISTS = ("There are some people whose views are considered extreme by the majority. Consider religious extremists, "
              "that is people who believe that their religion is the only true faith and all other religions should be "
              "considered as enemies. Do you think such people should be allowed to...")
BELIEVE = "Do you believe in"
ATTITUDE = "What is your personal attitude towards members of the following religious group?"
TRUE_FALSE = "Please indicate whether you think the following statement is true or false."
THREAT = "Do you consider those belonging to the following group as threatening or not threatening?"

SOURCE = {
    "v1": "If you were to consider your life in general these days, how happy or unhappy would you say you are, on the whole...",
    "v2": "All things considered, how satisfied are you with your relationships with family members?",
    "v3": "Do you think it is wrong or not wrong if a married person has sexual relations with someone other than his or her husband or wife, is it...",
    "v4": "And what about sexual relations between two adults of the same sex, is it...",
    "v5": "Do you personally think it is wrong or not wrong for a woman to have an abortion if the family has a very low income and cannot afford any more children.",
    "v6": f"{AGREE} A husband's job is to earn money; a wife's job is to look after the home and family.",
    "v7": f"{CONFIDENCE} The {{parliament}}",
    "v8": f"{CONFIDENCE} Business and industry",
    "v9": f"{CONFIDENCE} Churches and religious organizations",
    "v10": f"{CONFIDENCE} Courts and the legal system",
    "v11": f"{CONFIDENCE} Schools and the educational system",
    "v12": "How much do you agree or disagree with the following statement? Religious leaders should not try to influence how people vote in elections.",
    "v13": f"{AGREE} We trust too much in science and not enough in religious faith.",
    "v14": f"{AGREE} Looking around the world, religions bring more conflict than peace.",
    "v15": f"{AGREE} People with very strong religious beliefs are often too intolerant of others.",
    "v16": "Do you think that churches and religious organizations in this country have too much power or too little power?",
    "v17": "People have different religions and different religious views. Would you accept a person from a different religion or with a very different religious view from yours marrying a relative of yours.",
    "v18": f"{EXTREMISTS} Hold public meetings to express their views?",
    "v19": f"{EXTREMISTS} Publish their views on the Internet or social media?",
    "v20": "Please indicate which statement below comes closest to expressing what you believe about God.",
    "v21": "Which best describes your beliefs about God?",
    "v22": f"{BELIEVE} Life after death?",
    "v23": f"{BELIEVE} Heaven?",
    "v24": f"{BELIEVE} Hell?",
    "v25": f"{BELIEVE} Religious miracles?",
    "v26": f"{BELIEVE} The supernatural powers of deceased ancestors?",
    "v27": f"{AGREE} There is a God who concerns Himself with every human being personally.",
    "v28": f"{AGREE} There is little that people can do to change the course of their lives.",
    "v29": f"{AGREE} To me, life is meaningful only because God exists.",
    "v30": f"{AGREE} In my opinion, life does not serve any purpose.",
    "v31": f"{AGREE} Life is only meaningful if you provide the meaning yourself.",
    "v32": f"{AGREE} I have my own way of connecting with God without churches or religious services.",
    "v33": "In your opinion, does your religion treat men and women equally, treat men better than women, or treat women better than men?",
    "v35": "What was your mother's religious preference when you were a child?",
    "v37": "What was your father's religious preference when you were a child?",
    "v39": "What religion, if any, were you raised in?",
    "v40": "When you were a child, how often did your mother attend religious services?",
    "v41": "When you were a child, how often did your father attend religious services?",
    "v42": "And what about when you were around 11 or 12, how often did you attend religious services then?",
    "v43": "Now thinking about the present... About how often do you pray?",
    "v44": "How often do you take part in the activities or organizations of a church or place of worship other than attending services?",
    "v45": "During the last 12 months, have you read or listened to the reading of any holy scripture such as the Bible, Buddhist sutra, Koran, Sruti, Torah, or other religious scripture, not counting any reading that happened during a worship service?",
    "v46": "For religious reasons do you have in your home a shrine, altar, or a religious object on display such as an icon, retablo, mezuzah, menorah, or crucifix?",
    "v47": "How often do you visit a holy place for religious reasons such as going to a shrine, temple, church or mosque? Please do not count attending regular religious services at your usual place of worship, if you have one.",
    "v48": "Would you describe yourself as ...",
    "v49": "What best describes you?",
    "v50": "Do you agree or disagree that practicing a religion helps people to… make friends?",
    "v51": "Do you agree or disagree that practicing a religion helps people to… gain comfort in times of trouble or sorrow?",
    "v52": f"{AGREE} Religions are usually a barrier to equality between women and men.",
    "v53": f"{AGREE} Governments should not interfere with the attempts of any religion to spread its faith.",
    "v54": f"{AGREE} All things considered, people belonging to different religions cannot get along with each other when living close together.",
    "v55": f"{AGREE} In {{country}}, religion represents the past and not the future.",
    "v56": f"{AGREE} In {{country}}, religion is just as relevant to life today as it was in the past.",
    "v57": f"{ATTITUDE} Christians",
    "v58": f"{ATTITUDE} Muslims",
    "v59": f"{ATTITUDE} Hindus",
    "v60": f"{ATTITUDE} Buddhists",
    "v61": f"{ATTITUDE} Jews",
    "v62": f"{ATTITUDE} Atheists or non-believers",
    "v63": "In general, would you say your health is…",
    "v64": "Would you say that you have been \"born again\" or have had a \"born again\" experience -- that is, a turning point in your life when you committed yourself to Christ?",
    "v65": "Which of these statements comes closest to describing your feelings about the Bible?",
    "v66": "Has there ever been a turning point in your life when you made a new and personal commitment to religion?",
    "v67": f"{TRUE_FALSE} Good luck charms sometimes do bring good luck.",
    "v68": f"{TRUE_FALSE} Some fortune tellers really can foresee the future.",
    "v69": f"{TRUE_FALSE} Some faith healers do have God-given healing powers.",
    "v70": f"{TRUE_FALSE} A person's star sign at birth, or horoscope, can affect the course of their future.",
    "v71": "Suppose a law was passed which conflicted with your religious principles and teachings. Would you...",
    "v72": f"{THREAT} Christians",
    "v73": f"{THREAT} Muslims",
    "v74": f"{THREAT} Hindus",
    "v75": f"{THREAT} Buddhists",
    "v76": f"{THREAT} Jews",
    "v77": f"{THREAT} Non-believers",
    "v78": "Generally speaking, would you say that people can be trusted or that you can't be too careful in dealing with people?",
    "v79": f"{BELIEVE} Reincarnation – being reborn in this world again and again?",
    "v80": f"{BELIEVE} Nirvana?",
    "v81": f"{AGREE} A country's laws should not be based on any religion.",
}

CHOICE = {"v21", "v33", "v35", "v37", "v39", "v40", "v41", "v49", "v65", "v71"}
NOUL = {"v45", "v46", "v64", "v66"}

if __name__ == "__main__":
    write_question_bank(HERE, HERE / "raw" / "ZA7570_v2-1-0.dta", HERE / "raw" / "ZA7570_v2-1-0_missing.txt",
                        SOURCE, CHOICE, NOUL, {},
                        HIGH_FIRST + ("Complete ", "Far too much", "Yes, definitely", "Excellent", "People can almost always"))
