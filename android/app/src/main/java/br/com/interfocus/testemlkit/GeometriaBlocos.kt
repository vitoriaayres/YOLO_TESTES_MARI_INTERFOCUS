package br.com.interfocus.testemlkit

import kotlin.math.roundToInt

/*
 * Divisão da imagem em blocos para a segunda passagem. Faz o papel das caixas
 * que o Roboflow propunha no pipeline Python: sem detector, a imagem é varrida
 * numa grade 3x3 de blocos com metade do tamanho da imagem e 50% de sobreposição.
 * Assim, qualquer código menor que 1/4 da imagem cabe inteiro em algum bloco.
 */

private const val BLOCOS_POR_LADO = 3

fun gerarBlocos(largura: Int, altura: Int): List<Caixa> {
    val larguraBloco = largura / 2
    val alturaBloco = altura / 2
    val passoX = (largura - larguraBloco) / (BLOCOS_POR_LADO - 1)
    val passoY = (altura - alturaBloco) / (BLOCOS_POR_LADO - 1)

    return (0 until BLOCOS_POR_LADO).flatMap { linha ->
        (0 until BLOCOS_POR_LADO).map { coluna ->
            val esquerda = coluna * passoX
            val topo = linha * passoY
            Caixa(esquerda, topo, esquerda + larguraBloco, topo + alturaBloco)
        }
    }
}

// Margem em volta de um candidato: generosa no lado curto (o localizador às vezes
// corta as pontas das barras) e pequena no lado longo (zona de silêncio).
private const val MARGEM_LADO_CURTO = 0.5f
private const val MARGEM_LADO_LONGO = 0.05f

/**
 * Leva uma caixa achada na imagem reduzida para a foto original ([escala] =
 * foto / reduzida), com margem e sem passar das bordas da foto.
 */
fun expandirCaixa(caixa: Caixa, escala: Float, larguraFoto: Int, alturaFoto: Int): Caixa {
    val margem = minOf(caixa.largura, caixa.altura) * MARGEM_LADO_CURTO +
        maxOf(caixa.largura, caixa.altura) * MARGEM_LADO_LONGO
    return Caixa(
        esquerda = ((caixa.esquerda - margem) * escala).roundToInt().coerceAtLeast(0),
        topo = ((caixa.topo - margem) * escala).roundToInt().coerceAtLeast(0),
        direita = ((caixa.direita + margem) * escala).roundToInt().coerceAtMost(larguraFoto),
        base = ((caixa.base + margem) * escala).roundToInt().coerceAtMost(alturaFoto),
    )
}

/** Converte coordenadas de uma camada (bloco + margem branca + ampliação) para a foto original. */
data class Transformacao(val origemX: Int, val origemY: Int, val margem: Int, val escala: Float) {

    fun paraOriginal(caixa: Caixa) = Caixa(
        esquerda = converter(caixa.esquerda, origemX),
        topo = converter(caixa.topo, origemY),
        direita = converter(caixa.direita, origemX),
        base = converter(caixa.base, origemY),
    )

    private fun converter(valor: Int, origem: Int): Int = (valor / escala).roundToInt() - margem + origem
}
