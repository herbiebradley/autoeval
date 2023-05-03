import hydra
from omegaconf import OmegaConf

from autoeval import PersonaStatement, AgreementFilter, PersonaStatementChain
from autoeval.pipeline import get_model
from autoeval.configs import ModelConfig
from autoeval.constants import SRC_PATH


@hydra.main(
    config_path=str(SRC_PATH / "configs"),
    config_name="config",
    version_base="1.2",
)
def main(config):
    config = ModelConfig(**config)
    print("----------------- Config ---------------")
    print(OmegaConf.to_yaml(config.dict()))
    print("-----------------  End -----------------")
    print(config)
    persona = PersonaStatement(
        input_variables=["entity", "description", "preamble"],
        instruct=False,
    )
    llm = get_model(config=config)
    chain = PersonaStatementChain(llm=llm, prompt=persona)
    # TODO: wrapper around chain that takes n (number of generations per prompt?)
    result = chain.apply(
        [
            {
                "preamble": config.preamble,
                "entity": config.entity,
                "description": config.description,
            },
            {
                "preamble": config.preamble,
                "entity": config.entity,
                "description": config.description,
            },
        ]
    )
    print(len(result))
    for i in result:
        print(i["text"])
        print("-----------------")


if __name__ == "__main__":
    main()
