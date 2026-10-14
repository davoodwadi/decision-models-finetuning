"""Build the ISSP 2023 question bank: one row per item v1-v68 with the source-questionnaire wording, answer options from
the .dta value labels, and a decision-model question type. Output: data/issp2023/question_bank.csv

Wording is transcribed from the ISSP 2023 Final Source Questionnaire (raw/ZA10010_bq.pdf, August 2022), laid out for
self-completion. Form instructions ("Please, check one box on each line.") are left out; the answer options are given
separately. Placeholders: {country}, {nationality} (e.g. "American"), from data/issp_countries.csv, and
{association}: the regional association the optional Q38-Q43 ask about (build_dataset.py, ASSOCIATION).
Shared logic: data/issp_common.py.

Q1c asks about "[religion]", the dominant religion each country filled in; it is written here as "the dominant
religion in {country}"."""
import sys
from pathlib import Path

HERE = Path(__file__).parent                     # data/issp2023/
sys.path.insert(0, str(HERE.parent))
from issp_common import HIGH_FIRST, write_question_bank  # noqa: E402

TRULY = ("Some people say that the following things are important for being truly {nationality}. Others say they are not "
         "important. How important do you think the following is?")
AGREE = "How much do you agree or disagree with the following statement?"
PROUD = "How proud are you of {country} in the following?"
RELATIONS = f"Now we would like to ask a few questions about relations between {{country}} and other countries. {AGREE}"
IMMIGRANTS = ("There are different opinions about immigrants from other countries living in {country}. (By \"immigrants\" "
              f"we mean people who come to settle in {{country}}) {AGREE}")
CITIZEN = ("There are different opinions as to what it takes to be a good citizen. As far as you are concerned personally "
           "on a scale of 1 to 7, where 1 is not at all important and 7 is very important, how important is it:")
DEMOCRACY = "On the whole, on a scale of 0 to 10 where 0 is very poorly and 10 is very well."
RIGHTS = ("There are different opinions about people's rights in a democracy. On a scale of 1 to 7, where 1 is not at all "
          "important and 7 is very important, how important is it:")
EXTENT = "To what extent do you agree or disagree with the following statement?"

# v -> (source question number, wording)
SOURCE = {
    "v1": ("Q1a", f"{TRULY} To have been born in {{country}}."),
    "v2": ("Q1b", f"{TRULY} To have {{nationality}} citizenship."),
    "v3": ("Q1c", f"{TRULY} To be an adherent of the dominant religion in {{country}}."),
    "v4": ("Q1d", f"{TRULY} To respect {{nationality}} political institutions and laws."),
    "v5": ("Q1e", f"{TRULY} To feel {{nationality}}."),
    "v6": ("Q1f", f"{TRULY} To have {{nationality}} ancestry."),
    "v7": ("Q2a", f"{AGREE} I would rather be a citizen of {{country}} than of any other country in the world."),
    "v8": ("Q2b", f"{AGREE} There are some things about {{country}} today that make me feel ashamed of {{country}}."),
    "v9": ("Q2c", f"{AGREE} The world would be a better place if people from other countries were more like the {{nationality}} people."),
    "v10": ("Q2d", f"{AGREE} Generally speaking, {{country}} is a better country than most other countries."),
    "v11": ("Q2e", f"{AGREE} People should support their country even if the country is in the wrong."),
    "v12": ("Q3", "Thinking in general about nations and countries all over the world, to what extent do you agree or disagree with the following statement? It would be best if every nation had its own country where it could make its own decisions."),
    "v13": ("Q4", "Some people say that it is possible to become truly {nationality} if a person makes an effort. Others say a person has to be born {nationality} to be truly {nationality}. What is your position? A: It is possible to become truly {nationality} if a person makes an effort. B: A person has to be born {nationality} to be truly {nationality}."),
    "v14": ("Q5a", f"{PROUD} The way democracy works"),
    "v15": ("Q5b", f"{PROUD} Its political influence in the world"),
    "v16": ("Q5c", f"{PROUD} {{country}}'s economic achievements"),
    "v17": ("Q5d", f"{PROUD} Its achievements in sports"),
    "v18": ("Q5e", f"{PROUD} Its achievements in the arts and literature"),
    "v19": ("Q5f", f"{PROUD} Its history"),
    "v20": ("Q6a", f"{RELATIONS} {{country}} should limit the import of foreign products in order to protect its national economy."),
    "v21": ("Q6b", f"{RELATIONS} For certain problems, like environment pollution, international bodies should have the right to enforce solutions."),
    "v22": ("Q6c", f"{RELATIONS} {{country}} should follow its own interests, even if this leads to conflicts with other nations."),
    "v23": ("Q6d", f"{RELATIONS} Foreigners should not be allowed to buy land in {{country}}."),
    "v24": ("Q6e", f"{RELATIONS} {{country}}'s television should give preference to {{country}} films and programs."),
    "v25": ("Q6f", f"{RELATIONS} Large international companies are doing more and more damage to local businesses in {{country}}."),
    "v26": ("Q7a", f"{IMMIGRANTS} Immigrants increase crime rates."),
    "v27": ("Q7b", f"{IMMIGRANTS} Immigrants are generally good for {{country}}'s economy."),
    "v28": ("Q7c", f"{IMMIGRANTS} Immigrants take jobs away from people who were born in {{country}}."),
    "v29": ("Q7d", f"{IMMIGRANTS} Immigrants improve {{nationality}} society by bringing new ideas and cultures."),
    "v30": ("Q7e", f"{IMMIGRANTS} People born in {{country}} should be given preference over immigrants when it comes to jobs, housing, or health care."),
    "v31": ("Q8", "Do you think the number of immigrants to {country} nowadays should be…"),
    "v32": ("Q9", "Which of these statements about immigrants comes closest to your view:"),
    "v33": ("Q10", f"{CITIZEN} Never to try to evade taxes."),
    "v34": ("Q11", f"{CITIZEN} Always to obey laws and regulations."),
    "v35": ("Q12", f"{CITIZEN} To keep watch on the actions of government."),
    "v36": ("Q13", f"{CITIZEN} To help people in {{country}} who are worse off than yourself."),
    "v37": ("Q14", f"{CITIZEN} To be willing to serve in the military at a time of need."),
    "v38": ("Q15", f"{DEMOCRACY} How well does democracy work in {{country}} today?"),
    "v39": ("Q16", f"{DEMOCRACY} And what about 10 years ago? How well did democracy work in {{country}} then?"),
    "v40": ("Q17", f"{DEMOCRACY} And how about 10 years from now? How well do you think democracy will work in {{country}} then?"),
    "v41": ("Q18", f"{RIGHTS} That all citizens have an adequate standard of living."),
    "v42": ("Q19", f"{RIGHTS} That government authorities respect and protect the rights of minorities."),
    "v43": ("Q20", f"{RIGHTS} That people be given more opportunities to participate in public decision-making."),
    "v44": ("Q21", f"{RIGHTS} That citizens may engage in acts of civil disobedience when they oppose government actions."),
    "v45": ("Q22", f"Thinking about the situation in {{country}}. {EXTENT} It should be the government's responsibility to reduce income differences between the rich and the poor."),
    "v46": ("Q23", "In politics people sometimes talk of left and right. Where would you place yourself on a scale from 0 to 10 where 0 means the left and 10 means the right?"),
    "v47": ("Q24", f"{EXTENT} People like me don't have any say about what the government does."),
    "v48": ("Q25", f"{EXTENT} I don't think the government cares much what people like me think."),
    "v49": ("Q26", "Thinking of the public service in {country}, how committed is it to serve the people?"),
    "v50": ("Q27", "How widespread do you think corruption is in the public service in {country}?"),
    "v51": ("Q28", f"{EXTENT} The people, and not politicians, should make our most important policy decisions."),
    "v52": ("Q29", f"{EXTENT} Politicians talk too much and take too little action."),
    "v53": ("Q30", f"{EXTENT} There is a conflict between the ordinary people and those in power in {{country}}."),
    "v54": ("Q31", f"{EXTENT} Business leaders make their money at the expense of ordinary people."),
    "v55": ("Q32", f"{EXTENT} It is better to rely on the opinions of experts than the experiences of ordinary people."),
    "v56": ("Q33", "How often do you use the media, including television, newspapers, radio, and the internet, to get political news or information?"),
    "v57": ("Q34", f"{EXTENT} Media coverage about politics in {{country}} is biased."),
    "v58": ("Q35", "How often do you think that people would try to take advantage of you if they got the chance, and how often would they try to be fair?"),
    "v59": ("Q36", "Generally speaking, would you say that people can be trusted or that you can't be too careful in dealing with people?"),
    "v60": ("Q37", "Are you a citizen of {country}?"),
    "v61": ("Q38", "How much have you heard or read about {association}?"),
    "v62": ("Q39a", "Generally speaking, would you say that {country} benefits or does not benefit from being a member of {association}?"),
    "v63": ("Q39b", "Generally speaking, would you say that {country} would benefit or would not benefit from being a member of {association}?"),
    "v64": ("Q40", "How strongly do you agree or disagree with the following statement? {country} should follow {association} decisions, even if it does not agree with them."),
    "v65": ("Q41", "Generally, do you think that {association} should have... much more, more, as much, less, or much less power than the national governments of its member states?"),
    "v66": ("Q42", "If there were a referendum today to decide whether {country} does or does not become a member of {association}, would you vote in favour or would you vote against?"),
    "v67": ("Q43", "If there were a referendum today to decide whether {country} does or does not remain a member of {association}, would you vote in favour or would you vote against?"),
    "v68": ("Q44", f"{EXTENT} It would be best if all nations lived in their own countries so there are no minorities or immigrants."),
}

CHOICE = {"v32", "v62", "v63", "v66", "v67"}   # v62, v63: "Have never heard of ..." is outside the scale
NOUL = {"v60"}

if __name__ == "__main__":
    write_question_bank(HERE, HERE / "raw" / "ZA10010_v1-0-0.dta", HERE / "raw" / "ZA10010_v1-0-0_missing.txt",
                        {v: question for v, (_, question) in SOURCE.items()}, CHOICE, NOUL, {},
                        HIGH_FIRST + ("Essential", "Very committed", "People can almost always"))
