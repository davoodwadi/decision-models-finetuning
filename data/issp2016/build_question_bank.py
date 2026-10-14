"""Build the ISSP 2016 question bank (Role of Government V): one row per item v1-v63 with the source-questionnaire
wording (raw/ZA6900_bq.pdf, June 2015), answer options from the .dta value labels, and a question type.
Output: data/issp2016/question_bank.csv. Shared logic: data/issp_common.py."""
import sys
from pathlib import Path

HERE = Path(__file__).parent                     # data/issp2016/
sys.path.insert(0, str(HERE.parent))
from issp_common import HIGH_FIRST, write_question_bank  # noqa: E402

PROTEST = ("There are many ways people or organisations can protest against a government action they strongly oppose. "
           "Should the following be allowed?")
REVOLUTION = ("There are some people whose views are considered extreme by the majority. Consider people who want to "
              "overthrow the government by revolution. Do you think such people should be allowed to ...")
ECONOMY = ("Here are some things the government might do for the economy. Please show whether you are in favour of or "
           "against the following action.")
SPENDING = ("Please show whether you would like to see more or less government spending in the following area. Remember "
            "that if you say \"much more\", it might require a tax increase to pay for it.")
RESPONSIBILITY = "On the whole, do you think it should or should not be the government's responsibility to ..."
PROVIDE = "People have different opinions on who should provide services in {country}. Who do you think should primarily provide..."
INFLUENCE = ("Here is a list of people and organisations that can influence government actions: the media; trade unions; "
             "business, banks and industry; religious organisations/authorities; the military/army; organised crime; "
             "people who vote for the party/the parties in government; citizens in general; civic and voluntary "
             "organisations; international organizations (e.g. United Nations, International Monetary Fund).")
LIBERTIES = ("Now some questions about civil liberties and public security. Do you think that {country}'s government should "
             "or should not have the right to do the following:")
SECURITY = ("Some people think that governments should have the right to take certain measures in the name of national "
            "security. Others disagree. Do you think that {country}'s government should or should not have the right to "
            "do the following:")
TERROR = ("Suppose the government suspected that a terrorist act was about to happen. Do you think the authorities should "
          "have the right to…")
AGREE = "How much do you agree or disagree with the following statement?"
TAXES = ("Generally, how would you describe taxes in {country} today? We mean all taxes together, including wage "
         "deductions, income tax, taxes on goods and services and all the rest.")
TAX_AUTHORITIES = "In general, how often do you think that the tax authorities in {country} do the following…"
COMPANIES = "In general, how often do you think that major private companies in {country} do the following:"
SUCCESS = "How successful do you think the government in {country} is nowadays in the following area?"

SOURCE = {
    "v1": "In general, would you say that people should obey the law without exception, or are there exceptional occasions on which people should follow their consciences even if it means breaking the law?",
    "v2": f"{PROTEST} Organising public meetings to protest against the government",
    "v3": f"{PROTEST} Organising protest marches and demonstrations",
    "v4": f"{REVOLUTION} ... hold public meetings to express their views?",
    "v5": f"{REVOLUTION} ... publish books expressing their views?",
    "v6": "All systems of justice make mistakes, but which do you think is worse ... to convict an innocent person, or to let a guilty person go free?",
    "v7": f"{ECONOMY} Cuts in government spending",
    "v8": f"{ECONOMY} Government financing of projects to create new jobs",
    "v9": f"{ECONOMY} Less government regulation of business",
    "v10": f"{ECONOMY} Support for industry to develop new products and technology",
    "v11": f"{ECONOMY} Support for declining industries to protect jobs",
    "v12": f"{ECONOMY} Reducing the working week to create more jobs",
    "v13": f"{SPENDING} The environment",
    "v14": f"{SPENDING} Health",
    "v15": f"{SPENDING} The police and law enforcement",
    "v16": f"{SPENDING} Education",
    "v17": f"{SPENDING} The military and defence",
    "v18": f"{SPENDING} Old age pensions",
    "v19": f"{SPENDING} Unemployment benefits",
    "v20": f"{SPENDING} Culture and the arts",
    "v21": f"{RESPONSIBILITY} ... provide a job for everyone who wants one",
    "v22": f"{RESPONSIBILITY} ... keep prices under control",
    "v23": f"{RESPONSIBILITY} ... provide health care for the sick",
    "v24": f"{RESPONSIBILITY} ... provide a decent standard of living for the old",
    "v25": f"{RESPONSIBILITY} ... provide industry with the help it needs to grow",
    "v26": f"{RESPONSIBILITY} ... provide a decent standard of living for the unemployed",
    "v27": f"{RESPONSIBILITY} ... reduce income differences between the rich and the poor",
    "v28": f"{RESPONSIBILITY} ... give financial help to university students from low-income families",
    "v29": f"{RESPONSIBILITY} ... provide decent housing for those who can't afford it",
    "v30": f"{RESPONSIBILITY} ... impose strict laws to make industry do less damage to the environment",
    "v31": f"{RESPONSIBILITY} ... promote equality between men and women",
    "v32": f"{PROVIDE} Health care for the sick?",
    "v33": f"{PROVIDE} Care for older people?",
    "v34": f"{PROVIDE} School education for children?",
    "v35": f"{INFLUENCE} Which one do you think has the most influence on the actions of {{country}}'s government?",
    "v36": f"{INFLUENCE} Which one do you think has the second most influence on the actions of {{country}}'s government?",
    "v37": "Here are two opinions about what affects policies in {country}. Which of them comes closest to your view?",
    "v38": f"{LIBERTIES} ... keep people under video surveillance in public areas?",
    "v39": f"{LIBERTIES} ... monitor e-mails and any other information exchanged on the Internet?",
    "v40": "Here is a scale from 0 to 10 where 0 is \"all government information should be publicly available, even if this meant a risk to public security\" and 10 is \"public security should be given priority, even if this meant limiting access to government information\". Where would you place yourself on such a scale?",
    "v41": f"{SECURITY} collect information about anyone living in {{country}} without their knowledge",
    "v42": f"{SECURITY} collect information about anyone living in other countries without their knowledge",
    "v43": f"{TERROR} ... detain people for as long as they want without putting them on trial?",
    "v44": f"{TERROR} ... tap people's telephone conversations?",
    "v45": f"{TERROR} ... stop and search people in the street at random?",
    "v46": "Now some more questions about politics. How interested would you say you personally are in politics?",
    "v47": f"{AGREE} People like me don't have any say about what the government does",
    "v48": f"{AGREE} I feel that I have a pretty good understanding of the important political issues facing our country",
    "v49": f"{AGREE} People we elect as MPs try to keep the promises they have made during the election",
    "v50": f"{AGREE} Most civil servants can be trusted to do what is best for the country",
    "v51": f"{TAXES} First, for those with high incomes, are taxes ...",
    "v52": f"{TAXES} Next, for those with middle incomes, are taxes ...",
    "v53": f"{TAXES} Lastly, for those with low incomes, are taxes ...",
    "v54": f"{TAX_AUTHORITIES} … make sure people pay their taxes",
    "v55": f"{TAX_AUTHORITIES} … treat everyone in accordance with the law, regardless of their contacts or position in society?",
    "v56": f"{COMPANIES} … comply with laws and regulations?",
    "v57": f"{COMPANIES} … try to avoid paying their taxes?",
    "v58": "In your opinion, about how many politicians in {country} are involved in corruption?",
    "v59": "And in your opinion, about how many public officials in {country} are involved in corruption?",
    "v60": "In the last five years, how often have you or a member of your immediate family come across a public official who hinted they wanted, or asked for, a bribe or favour in return for a service?",
    "v61": f"{SUCCESS} Providing health care for the sick?",
    "v62": f"{SUCCESS} Providing a decent standard of living for the old?",
    "v63": f"{SUCCESS} Dealing with threats to {{country}}'s security?",
}

CHOICE = {"v1", "v6", "v32", "v33", "v34", "v35", "v36", "v37"}

if __name__ == "__main__":
    write_question_bank(HERE, HERE / "raw" / "ZA6900_v2-0-0.dta", HERE / "raw" / "ZA6900_v2-0-0_missing.txt",
                        SOURCE, CHOICE, (), {},
                        HIGH_FIRST + ("Strongly in favour", "Spend much more", "Quite successful"))
