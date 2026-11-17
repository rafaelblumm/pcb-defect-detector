from pathlib import Path
from statistics import median
from time import perf_counter
from typing import Optional

from IPython.display import display, display_markdown, Image
import numpy as np

from pcb_defect_detector import result_writer
from pcb_defect_detector.config import ImageRegistrationConfig, PCBDetectionPipelineConfigs, RegistrationAlgorithm
from pcb_defect_detector.image_registration import ImageRegistrator
from pcb_defect_detector.pipeline import PCBDetectionPipeline
from pcb_defect_detector.utils import load_image, load_image_from_url, to_gray


_IMAGE_REPOSITORY_URL = (
    "https://raw.githubusercontent.com/tangsanli5201/DeepPCB/"
    "08e98c4db5922613fb97176eb3d6497d48260cb1/PCBData/"
)
""" URL do repositório de imagens de PCBs """


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
    local_images_repo: Optional[Path],
    cfg: PCBDetectionPipelineConfigs = PCBDetectionPipelineConfigs()
) -> result_writer.ResultSet:
    """
    Executa o pipeline

    Args:
        group: ID do grupo de imagens
        img_id: ID da imagem
        local_images_repo: Caminho das imagens do repositório local. Se não
            informado, faz download do repositório remoto
    Return:
        Imagens resultantes do pipeline
    """
    pipeline = PCBDetectionPipeline(cfg)

    if local_images_repo is None:
        ref, target = download_image_set(group_id, img_id)
        return pipeline.process(ref, target)

    ref, target = resolve_image_paths(local_images_repo, group_id, img_id)
    return pipeline.load_and_process(ref, target)


def resolve_image_paths(
    local_images_repo: Path,
    group_id: str,
    img_id: str,
) -> tuple[Path, Path]:
    """
    Resolve caminho das imagens de teste

    Args:
        local_images_repo: Caminho das imagens do repositório local
        group: ID do grupo de imagens
        img_id: ID da imagem
    Returns:
        Tupla com caminho da imagem matriz e imagem de teste
    """
    base = Path(local_images_repo, f"group{group_id}", group_id)
    ref = Path(base, f"{img_id}_temp.jpg")
    target = Path(base, f"{img_id}_test.jpg")

    return (ref, target)


def display_results(rs: result_writer.ResultSet):
    """
    Exibe resultado formatado da execução do pipeline

    Args:
        rs: Resultados do pipeline
    """
    display_markdown("## Relatório de execução", raw=True)
    _display_final_result(rs)
    _display_timings(rs)
    _display_result_set(rs)


def _display_final_result(rs: result_writer.ResultSet):
    """
    Exibe resultado final do pipeline

    Args:
        rs: Resultados do pipeline
    """
    display_markdown(f"### Mousebites identificados", raw=True)
    display(Image(filename=rs.mousebite_path))


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
    display_markdown("### Imagens geradas no pipeline", raw=True)
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


def run_reg_algorithm_comparison(
    local_images_repo: Path,
    group_id: str,
    img_id: str,
    repeats: int
) -> dict:
    ref_path, target_path = resolve_image_paths(
        local_images_repo,
        group_id,
        img_id
    )
    ref = to_gray(load_image(ref_path))
    target = to_gray(load_image(target_path))

    registration_results = {}
    for algorithm in RegistrationAlgorithm:
        registrator = ImageRegistrator(
            ImageRegistrationConfig(algorithm=algorithm)
        )
        times = []
        for _ in range(repeats):
            start = perf_counter()
            dx, dy, metric = registrator.register(ref, target)
            times.append(perf_counter() - start)

        registration_results[algorithm] = {
            "median": median(times),
            "min": min(times),
            "dx": dx,
            "dy": dy,
            "metric": metric,
        }

    return registration_results


def display_reg_algorithm_comparison_table(registration_results: dict):
    """
    Exibe tabela com resultado da comparação entre algoritmos

    Args:
        registration_results: Resultados da execução
    """
    rows = [
        "| Algoritmo | Mediana (s) | Mínimo (s) | dx | dy | Métrica |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for algorithm, r in registration_results.items():
        rows.append(
            f"| {algorithm.value} | {r['median']:.3f} | {r['min']:.3f} | "
            f"{r['dx']} | {r['dy']} | {r['metric']:.4f} |"
        )

    ssd_median = registration_results[RegistrationAlgorithm.SSD]["median"]
    fft_median = registration_results[
        RegistrationAlgorithm.FFT_CROSS_CORRELATION
    ]["median"]
    rows.append("")
    rows.append(f"Razão de tempo SSD / FFT (mediana): **{ssd_median / fft_median:.2f}x**")

    display_markdown("\n".join(rows), raw=True)


def run_reg_algorithm_comparison_pipeline(
    local_images_repo: Path,
    group_id: str,
    img_id: str,
) -> dict:
    """
    Executa pipeline com dois algoritmos de registro translacional

    Args:
        local_images_repo: Caminho das imagens do repositório local
        group: ID do grupo de imagens
        img_id: ID da imagem
    Returns:
        Tupla com caminho da imagem matriz e imagem de teste
    """
    ref_path, target_path = resolve_image_paths(
        local_images_repo,
        group_id,
        img_id
    )
    pipeline_timings = {}
    for algorithm in RegistrationAlgorithm:
        pipeline_cfg = PCBDetectionPipelineConfigs(
            registration=ImageRegistrationConfig(algorithm=algorithm)
        )
        result_set = PCBDetectionPipeline(pipeline_cfg).load_and_process(
            ref_path,
            target_path
        )
        pipeline_timings[algorithm] = result_set.timings

    return pipeline_timings


def display_reg_algorithm_pipeline_comparison_table(pipeline_timings: dict):
    """
    Exibe tabela com resultado da comparação entre algoritmos com pipeline
    completo

    Args:
        pipeline_timings: Resultados da execução
    """
    rows = [
        "| Algoritmo | Registro (s) | Total (s) | % no registro |",
        "|---|---:|---:|---:|",
    ]
    for algorithm, timings in pipeline_timings.items():
        registration = timings["Registro de imagens"]
        total = sum(timings.values())
        rows.append(
            f"| {algorithm.value} | {registration:.3f} | {total:.3f} | "
            f"{100 * registration / total:.1f}% |"
        )

    display_markdown("\n".join(rows), raw=True)
