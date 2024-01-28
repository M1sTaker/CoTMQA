import json
from llamaapi import LlamaAPI

# Initialize the llamaapi with your api_token
llama = LlamaAPI("LL-Afw22QqBIyALT1N7FrPT7XtX84o6xTLnNZMADOttQxHS6hgayte7sTMQygk05ryp")

model_input = """
	   you are an extract machine, your task it to perform the following action: \
       extract a short answer of the question from the given the context directly without any analysis. \
       Use the following format: Answer:<answer>. \
           
       Here is the Question and Context. \
	   Question:Who harmed one of the people from the image?Already kown that one of the people from the image is Istanbul. \
       Context:Turkish Media Show 'Selfie' Video of Alleged Istanbul Suspect\nISTANBUL \u2014\u00a0\nTurkish media have broadcast a video of a man they say is the suspected gunman who killed 39 people during a New Year's attack on an Istanbul nightclub.\nThe video appears to be from a cell phone the man is carrying as he walks around Istanbul's Taksim square. He spends much of the time looking into the camera, but does not speak.\nIt is unclear when the video was recorded.\nAuthorities had earlier released a grainy image of the suspected shooter taken from security camera footage.\nTurkish media also reported Tuesday that authorities detained two foreign nationals at Istanbul's Ataturk airport in connection with the attack. No other details were given.\nTurkey's Deputy Prime Minister Numan Kurtulmus told reporters Monday that eight other people had been detained.\nA Turkish police handout picture of a suspect in Istanbul nightclub attack which killed at least 39 people on New Year's Eve, made available Jan. 2, 2017.\nThe attack began early Sunday with the gunman killing a police officer and a civilian outside the Reina nightclub before going inside. There were about 600 people, many of them foreigners, in the club at the time, some of whom jumped into the Bosphorus Strait in order to escape. Authorities said the shooter blended in with people leaving the club. In addition to those killed, about 70 people were injured.\nOn Tuesday, U.S. President Barack Obama spoke by phone with Erdogan to express his condolences. The White House said the two leaders agreed that Turkey and the United States must continue to stand united in order to defeat terrorism. Obama also praised Turkey's ongoing efforts to work with regional players to facilitate a nationwide cease-fire in Syria and a return to political negotiations between the Syrian regime and the opposition.\nA police officer looks at the photographs of the victims as two others stand guard a day after an attack at a popular nightclub in Istanbul, Jan. 2, 2017.\nIS claims responsibility \nIn a statement Monday, Islamic State said one of its \u201cheroic soldiers\u201d carried out the attack and that it targeted Turkey for siding \u201cwith countries of the cross.\" The group said the night club was targeted because it was a place where \u201cChristians celebrated their apostles.\u201d\n\"It was certainly expected that Islamic State would one way or another be linked to the attack,\u201d said political analyst Sinan Ulgen, of EDAM, an Istanbul-based political research group. \"Looking at both the nature of the target, a popular night club, (and) the timing, New Year\u2019s Eve, made it likely to be Islamic State.\u201d\n In a video released last week, Islamic State called on its supporters to launch attacks in Turkey. The video came against the backdrop of the Turkish military\u2019s ongoing battle to wrest control of the strategically important Syrian town of al-Bab from the jihadist group.\nCiting security sources, Turkish media reported Monday the gunman in the nightclub attack is believed to have come from a Central Asian country, either Kyrgyzstan or Uzbekistan. Many Islamic State fighters are drawn from Central Asian countries and have used Istanbul as a base before traveling to fight in Syria.\nExperts say Turkey is paying for the government\u2019s earlier Syrian policy. \u201cTurkey did choose to support Islamist-leaning groups of the Syrian rebel opposition, with the view and expectation that support would accelerate regime change in Syria,\u201d notes analyst Ulgen. \u201cWhat we have seen is these groups have taken advantage of the position of the Turkish government to set up (terror) cells within Turkey, which are now being used against Turkey.\u201c\nLocal media, citing a police report, said that three days before the nightclub attack, 63 suspected Islamic State militants were detained across Turkey, including in Istanbul. The same report said many of those held were from foreign countries and that the same jihadist cell which carried out June\u2019s attack on Istanbul\u2019s Ataturk airport could be behind this latest deadly attack.\nRelatives of Fatih Cakmak, a security guard and a victim of an attack by a gunman at Reina nightclub, react during his funeral in Istanbul, Turkey, Jan. 2, 2017.\nFunerals \nFunerals are continuing for those killed in the New Year's attack, which was the fourth deadly IS attack in Istanbul in a year.\nFor one unnamed mourner there was both anger and despair.\n\u201cThis is where words fail. What word, what sentence can depict this pain? After life is gone, the heart is gone. Terror is not only the problem of this country, it is the problem of every living person, but they just cannot find a solution,\u201d the mourner said.\nThe PKK Kurdish rebel group has also carried out bombings in Turkey's main cities, including last month in Istanbul. The government has promised to step up security.\nBut the security forces are struggling to cope with mass purges within their ranks after July\u2019s failed coup attempt, which Turkey's government has blamed on followers of the Islamic cleric Fethullah Gulen.\n
       """

# Define your API request
api_request_json = {
	"model": "llama-13b-chat",
	"messages": [
		{"role": "user", "content": f"{model_input}"},
	],
	"max_tokens": 50
}

# Make your request and handle the response
response = llama.run(api_request_json)

response_json = response.json()
content = response_json["choices"][0]["message"]["content"]

print(content)
# print(json.dumps(response.json(), indent=2))