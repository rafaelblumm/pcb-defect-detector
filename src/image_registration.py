import cv2
import numpy as np

from config import ImageRegistrationConfig, RegistrationAlgorithm


class ImageRegistrator:
    """
    Realiza registro translacional utilizando o algoritmo configurado:
    - SSD: busca local por soma da diferença dos quadrados (Sum of Squared
      Differences)
    - FFT_CROSS_CORRELATION: correlação cruzada calculada via Transformada
      Rápida de Fourier
    """

    def __init__(self, cfg: ImageRegistrationConfig):
        """
        Cria novo registrador de imagens

        Args:
            cfg: Configurações do registrador
        """
        self.cfg = cfg


    def register(
        self,
        reference_img: np.ndarray,
        target_img: np.ndarray
    ):
        """
        Realiza registro translacional da imagem resultado com a imagem matriz.

        Args:
            reference_img: Imagem referência
            target_img: Imagem alvo

        Returns:
            Tupla de (dx, dy, metric): Deslocamento horizontal, deslocamento
            vertical e métrica
        """
        if reference_img.shape != target_img.shape:
            raise ValueError(
                "Imagem de referência e imagem alvo possuem tamanhos "
                "diferentes"
            )

        algorithm = RegistrationAlgorithm(self.cfg.algorithm)
        if algorithm is RegistrationAlgorithm.FFT_CROSS_CORRELATION:
            return self._register_fft(reference_img, target_img)

        return self._register_ssd(reference_img, target_img)


    def _register_ssd(
        self,
        reference_img: np.ndarray,
        target_img: np.ndarray
    ):
        """
        Registro por busca local exaustiva com métrica SSD

        Args:
            reference_img: Imagem referência
            target_img: Imagem alvo

        Returns:
            Tupla de (dx, dy, ssd)
        """
        radius = self.cfg.radius
        displacements = [
            (dx, dy)
            for dx in range(-radius, radius + 1, self.cfg.step)
            for dy in range(-radius, radius + 1, self.cfg.step)
        ]

        best_displacement = (0, 0)
        best_metric = float('inf')

        for dx, dy in displacements:
            metric = self._compute_ssd_metric(
                reference_img,
                target_img,
                dx,
                dy
            )
            if metric < best_metric:
                best_metric = metric
                best_displacement = (dx, dy)

        return best_displacement[0], best_displacement[1], best_metric


    def _register_fft(
        self,
        reference_img: np.ndarray,
        target_img: np.ndarray
    ):
        """
        Registro por correlação cruzada calculada no domínio da frequência.
        As imagens são preenchidas com zeros para que a correlação seja linear
        (sem sobreposição circular). Para cada deslocamento dentro do raio
        configurado, a correlação é normalizada (ZNCC) apenas pela região de
        sobreposição, evitando favorecer deslocamentos pequenos, que têm mais
        área em comum. O pico define o deslocamento.

        Args:
            reference_img: Imagem referência
            target_img: Imagem alvo

        Returns:
            Tupla de (dx, dy, zncc), onde zncc é a correlação cruzada
            normalizada e centrada no pico, entre -1 e 1
        """
        height, width = reference_img.shape
        # Mantém ao menos metade da imagem sobreposta em cada eixo
        radius = min(self.cfg.radius, height // 2, width // 2)

        ref = reference_img.astype(np.float64)
        tgt = target_img.astype(np.float64)
        ones = np.ones_like(ref)

        # Preenchimento suficiente para evitar sobreposição circular nos
        # deslocamentos até o raio
        fft_shape = (height + radius, width + radius)
        offsets = np.arange(-radius, radius + 1)
        rows = offsets % fft_shape[0]
        cols = offsets % fft_shape[1]

        def rfft(img: np.ndarray) -> np.ndarray:
            return np.fft.rfft2(img, s=fft_shape)

        def correlate(a_fft: np.ndarray, b_fft: np.ndarray) -> np.ndarray:
            # out[dy, dx] = sum(a[y + dy, x + dx] * b[y, x]); deslocamentos
            # negativos ficam no final de cada eixo
            out = np.fft.irfft2(a_fft * np.conj(b_fft), s=fft_shape)
            return out[np.ix_(rows, cols)]

        ref_fft, tgt_fft, ones_fft = rfft(ref), rfft(tgt), rfft(ones)

        # Somas sobre a região de sobreposição de cada deslocamento
        count = correlate(ones_fft, ones_fft)
        sum_ref = correlate(ref_fft, ones_fft)
        sum_tgt = correlate(ones_fft, tgt_fft)
        sum_ref_sq = correlate(rfft(ref ** 2), ones_fft)
        sum_tgt_sq = correlate(ones_fft, rfft(tgt ** 2))
        sum_cross = correlate(ref_fft, tgt_fft)

        count = np.maximum(np.round(count), 1)
        cov = sum_cross - sum_ref * sum_tgt / count
        var_ref = sum_ref_sq - sum_ref ** 2 / count
        var_tgt = sum_tgt_sq - sum_tgt ** 2 / count

        # Variâncias muito pequenas (regiões uniformes) não têm correlação
        # confiável
        denom = np.sqrt(np.clip(var_ref, 0, None) * np.clip(var_tgt, 0, None))
        valid = denom > 1e-6 * count
        zncc = np.full(cov.shape, -np.inf)
        zncc[valid] = cov[valid] / denom[valid]

        if not valid.any():
            return 0, 0, 0.0

        peak_y, peak_x = np.unravel_index(np.argmax(zncc), zncc.shape)

        return (
            int(offsets[peak_x]),
            int(offsets[peak_y]),
            float(zncc[peak_y, peak_x])
        )


    def _compute_ssd_metric(
        self,
        reference_img: np.ndarray,
        target_img: np.ndarray,
        dx: int,
        dy: int,
    ):
        """
        Calcula métrica de diferença entre regiões da imagem

        Args:
            reference_img: Imagem referência
            target_img: Imagem alvo
            dx: Deslocamento horizontal
            dy: Deslocamento vertical

        Returns:
            Métrica de diferença entre regiões da imagem
        """
        height, width = reference_img.shape

        # Região de interesse baseado no deslocamento
        if dx >= 0:
            ref_x_start, ref_x_end = dx, width
            tgt_x_start, tgt_x_end = 0, width - dx
        else:
            ref_x_start, ref_x_end = 0, width + dx
            tgt_x_start, tgt_x_end = -dx, width

        if dy >= 0:
            ref_y_start, ref_y_end = dy, height
            tgt_y_start, tgt_y_end = 0, height - dy
        else:
            ref_y_start, ref_y_end = 0, height + dy
            tgt_y_start, tgt_y_end = -dy, height

        ref_roi = reference_img[ref_y_start:ref_y_end, ref_x_start:ref_x_end]
        tgt_roi = target_img[tgt_y_start:tgt_y_end, tgt_x_start:tgt_x_end]

        if ref_roi.size == 0 or tgt_roi.size == 0:
            return float('inf')

        return np.sum((ref_roi.astype(float) - tgt_roi.astype(float)) ** 2)


def apply_transformation(
        image: np.ndarray,
        dx: int,
        dy: int
    ) -> np.ndarray:
        """
        Aplica transformação translacional na imagem

        Args:
            image: Imagem de entrada
            dx: Deslocamento horizontal
            dy: Deslocamento vertical

        Returns:
            Imagem tranformada
        """
        return cv2.warpAffine(
            image,
            np.float32([[1, 0, dx], [0, 1, dy]]),
            (image.shape[0], image.shape[1])
        )
