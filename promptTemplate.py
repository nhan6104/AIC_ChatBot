from langchain_core.prompts import ChatPromptTemplate


EXTRACT_PROMPT_TEMPLATE = ChatPromptTemplate.from_messages([
        (
            "system",
            "Extract the number of distinct scenes described in the user's input. Extract and list each scene separately. Follow this template: num_of_scene: X \\n scene_1: [description] \\n scene_2: [description] \\n.... \  Only provide the number and the list of scenes without any additional commentary.",
        ),
        ("human", "{query}"),
    ])

ENRICH_PROMPT_TEMPLATE = ChatPromptTemplate.from_messages([
        (
            "system",
            """You are a semantic enrichment model.
            Given the following structured scene data, generate semantically rich, 
            natural language queries that describe the scene from different perspectives or intents.

            - Some queries should sound like a search intent (e.g., "Find drone footage of...")
            - Some should sound like visual descriptions (e.g., "A drone shot showing...")
            - Some should highlight themes or meanings (e.g., "Symbolizing urban development...")
            - All queries must remain faithful to the scene data.
            - Queries should add location or context that scene happened.(e.g., "Security camera footage and police bending down to scan happen in airports", "The barred posts in floodplains, ...")
            - Avoid overly generic queries; be specific to the scene details.

            Respond in English with only sentence. Avoid repeating the same structure. """
        ),
        ("human", "{query}"),
    ])


FEWSHOT_PROMPT_TEMPLATE = ChatPromptTemplate.from_messages([
        (
            "system",
            """You are an intelligent video reasoning assistant.
            Given a short scene description, infer the most likely real-world context or situation based on visual details, people, and objects.
            Use commonsense reasoning — for example, if there are flood markers, water, and police, the scene is likely in a flooded area under inspection.

            Expand the input into a natural, contextualized sentence describing the setting and what's happening.

            Examples:

            Example 1
            Input: “A police officer waves his hand for a person to come into the room.”
            Output: “At an airport security checkpoint, a police officer waves for the passenger to enter the inspection area.”

            Example 2
            Input: “The officer bends down to check something on the ground.”
            Output: “At an airport, the officer bends down to inspect a passenger's luggage during a security check.”

            Example 3
            Input: “Several people are standing around a stone pillar marked with numbers 10, 12, 14, 16, 18, and 20. Police officers are nearby. Later, the camera shows a river with fast-flowing water beneath a bridge.”
            Output: “The scene takes place in a flooded area where police officers are monitoring water levels. The videographer stands on a bridge over a fast-flowing river.”

            Example 4
            Input: “Two people in life jackets are near a rescue boat surrounded by floodwater.”
            Output: “Rescue workers are helping people in a flooded residential area.”

            Example 5
            Input: “A person is standing in front of a metal detector as another gestures to remove shoes.”
            Output: “At an airport security checkpoint, an officer instructs the passenger to remove their shoes for inspection.”

            Now your turn:
            Input: “[scene description]”
            Output: A natural, context-rich sentence describing the inferred situation.""",
        ),
        ("human", "{query}"),
    ])