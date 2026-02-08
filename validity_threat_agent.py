from google import genai
from google.genai import types
from dotenv import load_dotenv
import os

class ValidityThreatAgent:
    def __init__(self, api_key: str, model: str = "gemini-2.5-flash"):
        self.client = genai.Client(api_key=api_key)
        self.model = model

    def get_response(self, research_text: str) -> str:
        """
        Generate a response from the LLM.

        Args:
            research_text (str): The input.

        Returns:
            str: Generated response from the LLM.
        """

        # System instructions
        sys_instructions = f"""You are an expert research analyst. Analyze the given research text and identify potential validity threats.

Identify potential validity threats in the research text, categorizing them into internal, external, construct, and conclusion validity, based on the Validity Definition provided. For each category, provide a brief description of the identified threats. Additionally, suggest practical mitigation strategies for each identified threat to enhance the overall validity of the research.


\n--\n
Validity definition:

Given our objective to support the generation of hypotheses explaining a social system, a crucial question is whether the results of our analysis are valid. Validity concerns the relationship between our theory, theoretical concepts, and results,
and the system under study. This is a classic issue in social research, which
remains relevant when using digital data, albeit with some additional complexities.
Figure 1.6 can help clarify di!erent types of validity and should be used in practice
to identify the points where validity assessment is required. Conflating di!erent
parts of the figure is a common cause of validity issues. First, the system for which we want to produce a theory (System T , or ST in
short) may not be the same system we can observe. The observable system
(SO ) may be a subset of ST , for example when we only have data about part of
the population of interest, or it can just be similar to it, for example when we
have data about the whole population but only for a given time and we want
our theory to be valid over time. As a consequence, even when we have a solid
theory of SO , this does not guarantee external validity, that is, the applicability
of our theory to ST . Randomised controlled trials are examples where, through
random assignment to the treatment and control groups, we can obtain a high
internal validity (for SO ) but cannot conclude that the results will apply to ST
as well.
A second potential source of validity issues is that the data is not the same as
the observed system: it is not a neutral representation, as someone must have
decided which aspects to collect and which to omit, introducing biases. I will
call this data validity. Data can be organised as a set of observables measured
on a simple nominal or numerical scale, typically the domain of quantitative
social science and traditional statistics (e.g. hypothesis testing), or be more
unstructured, such as text and images. In this sense, the frequently used term raw data is something of an oxymoron: the
data (understood here as a model of the system at a certain level of abstraction,
following Floridi (2008)) is the result of explicit and implicit assumptions made
during the data collection process, and is, for example, constrained by meta-
model (the model defining how the data is structured). This could be something
like an Entity-Relationship conceptual meta-model, a relational meta-model, a data matrix, a network, text describing the system, or images/videos depicting
it. Data is never raw, it has always been designed and processed.
In addition to being shaped by the decisions involved in data collection and
constrained by its meta-model, the data may also su!er from quality issues: the
collection process itself can be imperfect, leading to inaccuracies, incomplete-
ness, and uncertainty. The di!erence between data and system is particularly
important to consider when the data is repurposed: collected for one purpose
and later used for research with a di!erent aim, which is a common task in the
area known as digital methods (Rogers, 2023).
This abstraction process can be repeated multiple times, as the initial data
(which is already a model of the observed system) may be too unorganised or
detailed to be considered a satisfactory set of patterns. For example, the data
can be transformed into a clustering model, here called a Machine Learning (ML)
model because it has been learned from the data using a computational method.
In this case, the entities in the data are grouped into newly defined concepts
(clusters). As before, the model (i.e., the clusters) is constrained by its meta-
model; for instance, choosing partitioning clustering means that overlapping
between clusters is not permitted. If there are naturally overlapping clusters in
the data, they will never be discovered by a partitioning method. Furthermore,
multiple models using the same meta-model can be learned from the same data,
leading to the question of which model is best. Model selection and optimisation
methods are used to address this question, but the answer may vary even when
using identical data. This will be a central type of validity when we discuss topic
models. ccuracy and related measures are often all that is reported in the ML literature,
sometimes together with execution times. This is an important part of the
concept of validity, because models that are too inaccurate are not valid. However,
accuracy is used quite di!erently in social data mining. The focus in the machine
learning literature is on accuracy maximisation. To compare the accuracy of
di!erent models, benchmarks are typically used, where a lot of algorithms are
tested against the same or a few datasets. On the contrary, in social data mining
the target systems are often di!erent in di!erent studies (Grimmer et al., 2021).
Therefore, because of limited external validity, we should not assume in general
that a reportedly more accurate algorithm (as assessed on benchmark data)
will be more accurate also on our data. Benchmark data, requiring to clearly
separate right from wrong, is also often di!erent from data used in social research,
especially when interpretive methods are used. In addition, as long as accuracy
is high enough to support our conclusions, other features of the algorithm such
as simplicity to use and transparency become more important. In fact, if well
documented, even accuracy that is too low to fully support our conclusions can
generate valuable hypotheses.
Finally, theory is often expressed in terms of concepts that are meaningful to
human beings, whereas machine learning and data mining models are data-driven,
so expressed in terms of variables in the data. Construct validity refers to the extent to which the entities in our models (that is, what can be measured)
correspond to the concepts in our theory. For example: do retweets (what we
measure) represent “endorsement” (a theoretical concept), “criticism”, or simply
“attention”? Do they indicate “influence”? Do clusters of social media accounts
formed by retweet patterns reflect a shared “ideology”, common “intentions”,
or “coordination”? These questions are essential, as our theories and interests
are usually centred on those high-level concepts (being more transferable across
domains and data), and not directly on what is measured in the data.

\n---\n

You MUST respond with ONLY a valid JSON object in this exact format:
{{
  "internal_validity": "description of internal validity threats or 'None identified'",
  "external_validity": "description of external validity threats or 'None identified'",
  "construct_validity": "description of construct validity threats or 'None identified'",
  "conclusion_validity": "description of conclusion validity threats or 'None identified'",
  "mitigation_suggestions": ["suggestion 1", "suggestion 2", "suggestion 3", ..., "suggestion n"]
}}

Do not include any text before or after the JSON. Avoid any markdown formatting.
        """

        # Generate a response from the llm
        response = self.client.models.generate_content(
            model=self.model,
            contents=research_text,
            config=types.GenerateContentConfig(
                temperature=0.0,
                system_instruction=sys_instructions
            )
        )
        response = response.text

        return response
    

# if __name__ == "__main__":
#     # Example usage
#     load_dotenv()
#     GOOGLE_API_KEY = os.getenv('GOOGLE_API_KEY')
#     validity_threat_agent = ValidityThreatAgent(api_key=GOOGLE_API_KEY)
#     user_input = "Example research text to analyze for validity threats."
#     response = validity_threat_agent.get_response(user_input)
#     print(response)
