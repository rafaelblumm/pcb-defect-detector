# pcb-defect-detector

Detector de defeitos em imagens de PCBs.

## Etapas do pipeline

1. Carrega imagem matriz e imagem resultado
2. Transforma imagens em escala de cinza
3. Alinha imagens
    - Usa algoritmo de SSD (Sum of Square Difference)
4. Calcula mapa de diferenças
    - Calcula diferença entre pixels
    - Aplica algoritmos de threshold
    - Filtra ruídos da imagem
5. Remover respostas espúrias (TO-DO)
6. Descrever geometricamente as regiões (TO-DO)
7. Indicar ocorrências sobre a imagem de teste
8. Produzir relatório com:
    - Localização
    - Dimensões
    - Confiança/Medida de resposta (TO-DO)
