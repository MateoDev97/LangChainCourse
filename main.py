from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_openai import ChatOpenAI

load_dotenv()


SUMMARY_TEMPLATE = """
Dada la información {information} acerca de la persona quiero que crees:
1. Un resumen corto
2. Dos datos interesantes acerca de él
"""


def summarize_with_openai(information: str) -> str:
    prompt = PromptTemplate.from_template(SUMMARY_TEMPLATE)
    llm = ChatOpenAI(model="gpt-5")
    response = (prompt | llm).invoke({"information": information})
    return response.content


def summarize_with_gemini(information: str) -> str:
    prompt = PromptTemplate.from_template(SUMMARY_TEMPLATE)
    llm = ChatGoogleGenerativeAI(model="gemini-3.6-flash", temperature=0)
    response = (prompt | llm).invoke({"information": information})
    return response.content


def main() -> None:
    information = """
    Marco Antonio Rubio (Miami, 28 de mayo de 1971) es un político y diplomático conservador estadounidense de ascendencia cubana, perteneciente al Partido Republicano. Es el secretario de Estado de los Estados Unidos desde el 21 de enero de 2025 y, como tal, el cuarto en la línea de sucesión presidencial del país, convirtiéndose en el primer político de ascendencia hispana en obtenerlo. Es también el primer político estadounidense en ser elegido para ocupar cargos públicos de forma simultánea: secretario de estado, consejero de seguridad nacional, administrador de USAID y archivista de los Estados Unidos.
    Hijo de inmigrantes cubanos, ha sido legislador de Florida, fue uno de los senadores en el Congreso de los Estados Unidos con raíces cubanas. Rubio fue precandidato del Partido Republicano para las elecciones presidenciales de Estados Unidos de 2016.
    """

    print(summarize_with_gemini(information))


if __name__ == "__main__":
    main()
