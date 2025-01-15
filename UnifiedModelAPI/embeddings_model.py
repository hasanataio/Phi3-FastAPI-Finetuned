
from angle_emb import AnglE

embeddings_model = AnglE.from_pretrained('WhereIsAI/UAE-Large-V1', pooling_strategy='cls').to("cuda")


async def get_embeddings(strs):
    result=embeddings_model.encode(strs)
    return result