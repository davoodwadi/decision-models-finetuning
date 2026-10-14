"""Build the ISSP 2024 question bank: one row per item v1-v78 with the source-questionnaire wording, answer options from
the .dta value labels, and a decision-model question type. Output: data/issp2024/question_bank.csv

Wording is transcribed from the ISSP 2024 Final Source Questionnaire (raw/ZA10020_bq.pdf, August 2023), laid out for
self-completion. Form instructions ("PLEASE TICK ONE BOX ONLY") are left out; the answer options are given separately.
Placeholders: {country}, {parliament} (the national legislature, from data/issp_countries.csv).
Shared logic: data/issp_common.py.

Some questions were asked only of part of the sample (Q1 is the filter). The wording here stays the questionnaire's; who
was asked goes into the persona (build_dataset.py, ASKED_OF).
"""
import sys
from pathlib import Path

HERE = Path(__file__).parent                     # data/issp2024/
sys.path.insert(0, str(HERE.parent))
from issp_common import HIGH_FIRST, write_question_bank  # noqa: E402

INTERNET = ("The following questions are about your Internet use. This includes activities such as communicating, buying "
            "things, searching for information, entertainment which you do through devices like computers, tablets, "
            "smartphones, mobile phones or other devices connected in some way to the Internet.")
REASONS = "Is the following a reason you have not used the Internet in the last 12 months?"
LOCATION = "How often, if at all, do you typically use the Internet on any device at the following location for any reason?"
SKILLS = "How would you rate your ability to do the following?"
PURPOSES = "How often, if at all, do you use the Internet for the following purpose?"
AGREE = "How much do you agree or disagree with the following statement?"
VALIDATION = ("When looking for news or information about political matters online, how often, if at all, do "
              "you do the following?")
POLARIZATION = ("In the last 12 months, if at all, has your online contact with the following people increased, "
                "decreased or remained the same?")
TECHNOLOGY = ("Here are some statements about technology. Technology includes all kinds of machinery, equipment, digital "
              "devices, and software. How much do you agree or disagree with this statement?")
RELIABLE = "In general, how reliable would you rate the news on political matters found in the following source?"
ACTION = ("Here are some different forms of political and social action that people can take. Have you done the "
          "following in the last 12 months offline only, online only, or both offline and online?")
TRUST = ("On a scale of 0 to 10, how much do you personally trust each of the following institutions? 0 means you do not "
         "trust an institution at all, and 10 means you trust it completely.")
GOVERNMENT = "Do you think that {country}'s government should or should not have the right to do the following?"
COMPANIES = ("Private companies offer services in return for personal information such as your name, phone number, email "
             "address, online-search history, or purchases. Regardless of whether you use the Internet yourself, please "
             "indicate the extent to which you agree or disagree with the following statement.")
BENEFIT = ("Do you think that in {{country}} {} are benefitting more, or both equally, or neither, from the use of the "
           "Internet and digital technology?")
AI = ("The next questions deal with robots and artificial intelligence. 'Artificial intelligence' is the ability of a "
      "robot or computer to perform tasks commonly associated with humans.")
FEEL = (f"{AI} Various human actions can be done by robots or Artificial Intelligence (AI). Using a scale from 0 to 10, "
        "how would you personally feel about the following? '0' means that you would feel \"totally uncomfortable\" and "
        "'10' means that you would feel \"totally comfortable\" with this situation.")

# v -> (source question number, wording)
SOURCE = {
    "v1": ("Q1", f"{INTERNET} During the last 12 months, how often, if at all, did you typically use the Internet on any device whether for work or personal use? If you are uncertain about your usage, please select the answer that comes closest to your use."),
    "v2": ("Q2a", f"{REASONS} The Internet is not available to me."),
    "v3": ("Q2b", f"{REASONS} The Internet connection is poor."),
    "v4": ("Q2c", f"{REASONS} The Internet is too expensive."),
    "v5": ("Q2d", f"{REASONS} I worry about Internet security and privacy."),
    "v6": ("Q2e", f"{REASONS} The Internet is not important or useful for me."),
    "v7": ("Q2f", f"{REASONS} The Internet is not for people of my age."),
    "v8": ("Q2g", f"{REASONS} The Internet is difficult for me to use."),
    "v9": ("Q2h", f"{REASONS} I do not want to use the Internet."),
    "v10": ("Q2i", f"{REASONS} I have difficulty reading or writing."),
    "v11": ("Q2j", f"{REASONS} I am too busy to use the Internet."),
    "v12": ("Q2k", f"{REASONS} The authorities restrict access to the Internet in {{country}}."),
    "v13": ("Q2l", f"{REASONS} Other reasons"),
    "v14": ("Q3", f"In the last 12 months how often, if at all, did you ask someone to do something for you on the Internet on your behalf, such as sending an email, getting information or shopping online?"),
    "v15": ("Q4", f"If you needed to do something on the Internet in the near future, how likely is it that someone will do it for you?"),
    "v16": ("Q5", f"How likely is it that you will personally use the Internet in the next 12 months?"),
    "v17": ("Q6a", f"{LOCATION} At home"),
    "v18": ("Q6b", f"{LOCATION} At your work-place/ at school/ at university"),
    "v19": ("Q6c", f"{LOCATION} In public places"),
    "v20": ("Q7", f"In the last 12 months, how often, if at all, were you asked by someone to do something on their behalf on the Internet, such as sending an email, getting information or shopping online? Exclude work related tasks."),
    "v21": ("Q8", f"In the last 12 months, how often, if at all, did you deliberately reduce the time you spent on the Internet?"),
    "v22": ("Q9", f"What is the main reason why you deliberately reduced the time you spent on the Internet?"),
    "v23": ("Q10a", f"{SKILLS} Using the Internet"),
    "v24": ("Q10b", f"{SKILLS} Doing online searches"),
    "v25": ("Q10c", f"{SKILLS} Knowing which information should not be shared online"),
    "v26": ("Q10d", f"{SKILLS} Learning how to use new software or apps"),
    "v27": ("Q11a", f"{PURPOSES} Communicating (for example writing, talking, video calling)"),
    "v28": ("Q11b", f"{PURPOSES} Sharing or posting photos or videos"),
    "v29": ("Q11c", f"{PURPOSES} Searching for information"),
    "v30": ("Q11d", f"{PURPOSES} Playing video games"),
    "v31": ("Q11e", f"{PURPOSES} Downloading or listening to music or watching videos and movies"),
    "v32": ("Q11f", f"{PURPOSES} Shopping, paying bills and other financial transactions"),
    "v33": ("Q12a", f"{AGREE} I feel more comfortable meeting with people online than in person."),
    "v34": ("Q12b", f"{AGREE} Without the Internet I would feel lonely."),
    "v35": ("Q13a", f"{VALIDATION} Read, watch or listen to something online you disagree with"),
    "v36": ("Q13b", f"{VALIDATION} Check a news source that's different from what you normally read, watch or listen to online"),
    "v37": ("Q13c", f"{VALIDATION} Try to confirm information you find by searching online for another source"),
    "v38": ("Q14", f"On a scale of 0 to 10, how much do you trust people you are communicating with on the Internet but have never met in person? 0 means you do not trust them at all, and 10 means you trust them completely."),
    "v39": ("Q15a", f"{POLARIZATION} People who share your political views"),
    "v40": ("Q15b", f"{POLARIZATION} People with different political views from your own"),
    "v41": ("Q16", f"How often, if at all, did you express, support or oppose political views on the Internet or social media in the last 12 months?"),
    "v42": ("Q17a", f"{TECHNOLOGY} Technology makes our lives easier."),
    "v43": ("Q17b", f"{TECHNOLOGY} Overall, technology does more harm than good."),
    "v44": ("Q17c", f"{TECHNOLOGY} Technology provides more opportunities for the next generation."),
    "v45": ("Q18", "Some people receive emails or text messages from scammers from time to time, some of them also become victims. Apart from only receiving such messages, have you or someone you know personally ever been a victim of an online scam, identity theft or hacking attack?"),
    "v46": ("Q19", "Some people report experiences of harassment on the Internet. Have you or someone you know personally ever experienced online harassment or online hate speech?"),
    "v47": ("Q20", "Where do you get most of your news about political matters? Please select the most important source."),
    "v48": ("Q21a", f"{RELIABLE} Online social media (for example WhatsApp, Facebook, Tiktok, Instagram, Youtube, Twitter, Telegram)"),
    "v49": ("Q21b", f"{RELIABLE} Television (including TV apps or access via Internet)"),
    "v50": ("Q21c", f"{RELIABLE} Newspapers (in print or access via Internet)"),
    "v51": ("Q21d", f"{RELIABLE} Radio and podcasts (including Internet radio or access via Internet)"),
    "v52": ("Q21e", f"{RELIABLE} Websites (including exclusively online news sites, news aggregators, portals, or search engines)"),
    "v53": ("Q22", "How interested would you say you personally are in politics?"),
    "v54": ("Q23a", f"{ACTION} Signed a petition"),
    "v55": ("Q23b", f"{ACTION} Took part in a demonstration or political protest"),
    "v56": ("Q23c", f"{ACTION} Contacted, or attempted to contact, a politician to express your views"),
    "v57": ("Q23d", f"{ACTION} Organised or helped to organise a demonstration or political protest"),
    "v58": ("Q23e", f"{ACTION} Belonged to or joined a group that is involved in social issues or working to advance a cause"),
    "v59": ("Q24", "In politics people sometimes talk of left and right. Where would you place yourself on a scale from 0 to 10 where 0 means the left and 10 means the right?"),
    "v60": ("Q25", "Generally speaking, would you say that most people can be trusted or that you can't be too careful in dealing with people? Please tick one box to show what you think, where 0 means you can't be too careful, and 10 means most people can be trusted."),
    "v61": ("Q26a", f"{TRUST} The {{parliament}}"),
    "v62": ("Q26b", f"{TRUST} {{country}}'s courts"),
    "v63": ("Q27", "Who should be mostly responsible for protecting personal information when using the Internet?"),
    "v64": ("Q28a", f"{GOVERNMENT} Keep people under video surveillance in public areas?"),
    "v65": ("Q28b", f"{GOVERNMENT} Monitor e-mails, social media content and any other personal information exchanged on the Internet?"),
    "v66": ("Q29a", f"{COMPANIES} I would provide my personal information online in return for discounts and free products."),
    "v67": ("Q29b", f"{COMPANIES} I would provide my personal information online even though private companies may use it to make a profit."),
    "v68": ("Q30", "In all countries, there are differences between social groups. " + BENEFIT.format("women or men")),
    "v69": ("Q31", BENEFIT.format("older people or younger people")),
    "v70": ("Q32", BENEFIT.format("people who live in the countryside or people who live in cities")),
    "v71": ("Q33", BENEFIT.format("the highly educated or the less well educated")),
    "v72": ("Q34", BENEFIT.format("rich people or poor people")),
    "v73": ("Q35", f"{AI} Overall, how worried, if at all, are you that in the next 10 years machines, computer programmes, and Artificial Intelligence (AI) will take over many of the jobs done by humans?"),
    "v74": ("Q36a", f"{FEEL} Having a medical operation performed on you by a robot"),
    "v75": ("Q36b", f"{FEEL} Being driven in a driverless car"),
    "v76": ("Q36c", f"{FEEL} Not being able to tell whether information is from an Artificial Intelligence (AI) or a real person."),
    "v77": ("Q36d", f"{FEEL} Having to communicate with an Artificial Intelligence (AI) instead of a real person."),
    "v78": ("Q36e", f"{FEEL} Not being able to check the accuracy of information generated by an Artificial Intelligence (AI)."),
}

CHOICE = {"v17", "v18", "v19", "v22", "v45", "v46", "v47", "v54", "v55", "v56", "v57", "v58", "v63", "v68", "v69", "v70", "v71", "v72"}
NOUL = {f"v{i}" for i in range(2, 14)}           # Q2: yes / no
LOW_FIRST = {"v48", "v49", "v50", "v51", "v52"}   # "Very unreliable" ... "Very reliable" already runs low -> high

if __name__ == "__main__":
    write_question_bank(HERE, HERE / "raw" / "ZA10020_v1-0-0.dta", HERE / "raw" / "ZA10020_v1-0-0_missing.txt",
                        {v: question for v, (_, question) in SOURCE.items()}, CHOICE, NOUL, {}, HIGH_FIRST, LOW_FIRST)
