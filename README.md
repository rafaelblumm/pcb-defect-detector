# pcb-defect-detector

Detector de defeitos em imagens de PCBs.

## Etapas do pipeline

1. Carrega imagem matriz e imagem resultado
2. Transforma imagens em escala de cinza
3. Alinha imagens
    - Usa algoritmo de SSD (Sum of Square Difference)
4. Calcula mapa de diferenças e identifica regiões pelo limiar
    - Calcula diferença entre pixels
    - Aplica algoritmos de threshold
    - Filtra ruídos da imagem
5. Identifica defeitos
    - Compara presença de cobre na referência e na imagem sendo validada
6. Categoriza defeitos
    - Analisa contexto adjacente
        - Borda da trilha
        - Trechos de cobre adjacentes
    - Categoriza mouse bite. Demais defeitos são categorizados como "desconhecido"
7. Grava arquivos de imagens
    1. Imagem registrada: `registered.png`
    2. Imagem com diferença: `difference.png`
    3. Imagem com limiar: `threshold.png`
    4. Imagem com todos defeitos: `defects.png`
    5. Imagem com defeitos filtrados: `classified_defects.png`

8. Produzir relatório com:
    - Localização
    - Dimensões
    - Confiança/Medida de resposta (TO-DO)
