from pathlib import Path

from IPython.display import display, display_markdown, Image
import numpy as np

from config import PCBDetectionPipelineConfigs
from pipeline import PCBDetectionPipeline
import result_writer
from utils import load_image_from_url


_IMAGE_REPOSITORY_URL = (
    "https://raw.githubusercontent.com/tangsanli5201/DeepPCB/"
    "08e98c4db5922613fb97176eb3d6497d48260cb1/PCBData/"
)
""" URL do repositório de imagens de PCBs """

# LOCAL_REPOSITORY_PATH = Path("/path/to/repo/DeepPCB/PCBData")
LOCAL_REPOSITORY_PATH = Path("/home/blumm/Documents/dev/DeepPCB/PCBData")
""" Caminho do repositório local de imagens """


def download_image_set(group: str, img_id: str) -> (np.ndarray):
    """
    Carrega imagens a partir de uma URL

    Args:
        group: ID do grupo de imagens
        img_id: ID da imagem
    Returns:
        Tupla com imagem referência e imagem de teste
    """
    ref_url = f"{_IMAGE_REPOSITORY_URL}/group{group}/{group}/{img_id}_temp.jpg"
    print("Fazendo download da imagem referência")
    print(f"  URL: {ref_url}")
    ref = load_image_from_url(ref_url)

    test_url = f"{_IMAGE_REPOSITORY_URL}/group{group}/{group}/{img_id}_test.jpg"
    print("Fazendo download da imagem de teste")
    print(f"  URL: {test_url}")
    test = load_image_from_url(test_url)

    return (ref, test)


def run_pipeline(
    group_id: str,
    img_id: str,
    from_local_fs: bool = True,
    cfg: PCBDetectionPipelineConfigs = PCBDetectionPipelineConfigs()
) -> result_writer.ResultSet:
    """
    Executa o pipeline

    Args:
        group: ID do grupo de imagens
        img_id: ID da imagem
        from_local_fs: Se usa imagens do repositório local. Se falso, faz
            download do repositório remoto
    Return:
        Imagens resultantes do pipeline
    """
    pipeline = PCBDetectionPipeline(cfg)

    if not from_local_fs:
        ref, target = download_image_set(group_id, img_id)
        return pipeline.process(ref, target)

    base = Path(LOCAL_REPOSITORY_PATH, f"group{group_id}", group_id)
    ref = Path(base, f"{img_id}_temp.jpg")
    target = Path(base, f"{img_id}_test.jpg")
    return pipeline.load_and_process(ref, target)


def display_results(rs: result_writer.ResultSet):
    """
    Exibe resultado formatado da execução do pipeline

    Args:
        rs: Resultados do pipeline
    """
    display_markdown("## Relatório de execução", raw=True)
    _display_timings(rs)
    _display_result_set(rs)


def _display_timings(rs: result_writer.ResultSet):
    """
    Exibe tabela com o tempo de execução de cada etapa do pipeline

    Args:
        rs: Resultados do pipeline
    """
    total = sum(rs.timings.values())
    rows = [
        "| Etapa | Tempo (s) | % do total |",
        "|---|---:|---:|"
    ]
    for stage, seconds in rs.timings.items():
        pct = 100 * seconds / total
        rows.append(f"| {stage} | {seconds:.3f} | {pct:.1f}% |")
    rows.append(f"| **Total** | **{total:.3f}** | **100%** |")

    display_markdown("### Tempo de execução por etapa", raw=True)
    display_markdown("\n".join(rows), raw=True)


def _display_result_set(rs: result_writer.ResultSet):
    """
    Exibe resultado no Jupyter Notebook

    Args:
        rs: Resultados do pipeline
    """
    display_markdown("### Resultado do pipeline", raw=True)
    fmt_results = [
        ("0. Imagem matriz", rs.ref_path),
        ("1. Imagem registrada", rs.registered_path),
        ("2. Imagem de diferença", rs.diff_path),
        ("3. Imagem de limiar", rs.threshold_path),
        ("4. Imagem com defeitos identificados", rs.defects_path),
        ("5. Imagem com defeitos mousebite identificados", rs.mousebite_path)
    ]
    for title, img_path in fmt_results:
        display_markdown(f"#### {title}", raw=True)
        display(Image(filename=img_path))

