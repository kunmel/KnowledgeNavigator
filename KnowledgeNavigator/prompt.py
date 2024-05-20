import config
SIMILAR_QUERY_PROMPT = """<s>[INST] <<SYS>>
Based on the following question, rewrite {number} questions with the same meaning. You only need to rewrite the questions and do not add any additional information. Your response should only include the questions you rewrite like Q1:[question1], Q2:[question2]... .
<</SYS>>

Your task:
Question: {query}
Answer: [/INST]"""

RELATION_RETRIEVE_PROMPT = """<s>[INST] <<SYS>>
Please consider all of the information below and select {relation_number} relations connected to the core entity that are most relevant to the target question from the relations provided below. Your response should only include the name of the relation you choose like R1:[relation1], R2:[relation2].
<</SYS>>

Target question: {query}
Core entity: {entity}
Relations connected to the entity: {relations}
Answer: [/INST]"""

ANSWER_GENERATE_PROMPT_HITS1 = """<s>[INST] <<SYS>>
You are a Q&A robot that answers questions by summarizing the knowledge provided, but do not use the knowledge not provided bellow. If the question has multiple correct answers, you only need to provide one of the answers in the format: "A:".
<</SYS>>

Target question: {query}
Knowledge: {knowledge}
Answer: [/INST]"""

def get_similar_question_prompt(number, query):
    return SIMILAR_QUERY_PROMPT.format(number=number, query=query)

def get_relation_retrieve_prompt(query, relations, relation_number, entity):
    return RELATION_RETRIEVE_PROMPT.format(query=query, relations=relations, relation_number=relation_number, entity=entity)

def get_answer_prompt(query, knowledge):
    return ANSWER_GENERATE_PROMPT_HITS1.format(query=query, knowledge=knowledge)


RELATION_RETRIEVE_PROMPT_FEWSHOT_METAQA = """<s>[INST] <<SYS>>
Please consider all of the information below and select {relation_number} relations connected to the core entity that are most relevant to the target question from the relations provided below. Note that your response should contain the full name of the relations you selected, no missing characters. Your response should only include the name of the relation you choose without description like R1:[relation1], R2:[relation2].
Examples)
Example 1: 
Target question: the films that share actors with the film Dil Chahta Hai were released in which years?
Core entity: Dil Chahta Hai
Relations connected to the entity: in_language; release_year; written_by; starred_actors; has_tags; has_genre; directed_by
Answer: R1:starred_actors

Example 2: 
Target question: who are the directors of the movies written by the writer of The Green Mile?
Core entity: The Green Mile
Relations connected to the entity: directed_by; starred_actors; has_tags; release_year; has_genre; written_by
Answer: R1:written_by
<</SYS>>
Your Task)
Target question: {query}
Core entity: {entity}
Relations connected to the entity: {relations}
Answer: [/INST]"""

ANSWER_GENERATE_PROMPT_HITS1_FEWSHOT_METAQA = """<s>[INST] <<SYS>>
You are a Q&A robot that answers questions by summarizing the knowledge provided, but do not use the knowledge not provided bellow. If the question has multiple correct answers, you only need to provide one of the answers in the format: "A:". Be careful not to miss any letters in the words.
Example)
Example 1:
Target question: what were the release dates of films starred by actors in Caramel?
Knowledge: Nadine Labaki acted in Caramel, Where Do We Go Now?. The movie Where Do We Go Now? is released in 2011. 
Answer: A:2011

Example 2:
Target question: what are the primary languages in the movies written by the writer of Ugly?
Knowledge: The movies written by Anurag Kashyap include Ugly, Water, Udaan. The movie Udaan is in Hindi.
Answer: A:Hindi
<</SYS>>
Your Task)
Target question: {query}
Knowledge: {knowledge}
Answer: [/INST]"""


RELATION_RETRIEVE_PROMPT_FEWSHOT_WEBQSP = """<s>[INST] <<SYS>>
Please consider all of the information below and select {relation_number} relations connected to the core entity that are most relevant to the target question from the relations provided below. Note that your response should contain the full name of the relations you selected, no missing characters. Your response should only include the name of the relation you choose without description like R1:[relation1], R2:[relation2].
Example)
Example 1:
Target question: what kind of money to take to bahamas?
Core entity: Bahamas
Relations connected to the entity:  statistical_region.gdp_nominal_per_capita; flag_referent.flag; locations.countries.continent; country.first_level_divisions; organization_scope.organizations_with_this_scope; linguistic_hint.adjectival_form; country.capital; country.currency_used; country.form_of_government; country.national_anthem; location.adjectival_form; olympic_participating_country.medals_won; topic.alias; topic.article; 
Answer: R1:country.currency_used

Example 2:
Target question: where are the nfl redskins from?
Core entity: Washington Redskins
Relations connected to the entity: topic.topic_equivalent_webpage; topic.topical_webpage; topic.webpage; object_profile.prominent_type; object.key; topic.official_website; sports_team.founded; topic.image; sports_team.league; topic.social_media_presence; organization.founders; sports_team.location; person_or_entity_appearing_in_film.films; employer.employees; identity.daylife_topic; organization.leadership; avh.default_domain.ellerdale_topic.ellerdale_id; organization.previous_names; 
Answer: R1:sports_team.location
<</SYS>>

Your task)
Target question: {query}
Core entity: {entity}
Relations connected to the entity: {relations}
Answer: [/INST]"""

ANSWER_GENERATE_PROMPT_HITS1_FEWSHOT_WEBQSP = """<s>[INST] <<SYS>>
You are a Q&A robot that answers questions by summarizing the knowledge provided, but do not use the knowledge not provided bellow.If the question has multiple correct answers, you only need to provide one of the answers in the format: "A:". Be careful not to miss any letters in the words.
Example)
Example 1:
Target question: what kind of money to take to bahamas?
Knowledge: The currency_used of Bahamas are: Bahamian dollar.
Answer: A:Bahamian dollar

Example 2:
Target question: where are the nfl redskins from?
Knowledge: The location of Washington Redskins are: Washington, D.C.
Answer: A:Washington, D.C.
<</SYS>>

Your task)
Target question: {query}
Knowledge: {knowledge}
Answer: [/INST]"""


def get_relation_retrieve_prompt_fewshot(query, relations, relation_number, entity):
    if config.TASK == "METAQA": 
        return RELATION_RETRIEVE_PROMPT_FEWSHOT_METAQA.format(query=query, relations=relations, relation_number=relation_number, entity=entity)
    else:
        return RELATION_RETRIEVE_PROMPT_FEWSHOT_WEBQSP.format(query=query, relations=relations, relation_number=relation_number, entity=entity)
        

def get_answer_prompt_fewshot(query, knowledge):
    if config.TASK == "METAQA":
        return ANSWER_GENERATE_PROMPT_HITS1_FEWSHOT_METAQA.format(query=query, knowledge=knowledge)
    else:
        return ANSWER_GENERATE_PROMPT_HITS1_FEWSHOT_WEBQSP.format(query=query, knowledge=knowledge)
    
