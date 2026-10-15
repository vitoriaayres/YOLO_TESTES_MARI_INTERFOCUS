package br.com.interfocus.testemlkit

import kotlin.math.pow
import kotlin.math.roundToInt

/*
 * Filtros das camadas 3 a 5, portados de barcode_layer_pipeline (barcode_pipeline.py).
 * Trabalham sobre tons de cinza puros, sem Android, para poder ser testados com JUnit.
 */

private const val INTENSIDADE_NITIDEZ = 0.35 // super_resolution_sharpen
private const val PERCENTIL_CORTE_CONTRASTE = 0.01 // ignora 1% dos pixels extremos de cada lado

/** Imagem em tons de cinza (0..255), um valor por pixel, linha a linha. */
class ImagemCinza(val largura: Int, val altura: Int, val pixels: IntArray) {
    init {
        require(pixels.size == largura * altura) {
            "Esperado ${largura * altura} pixels, recebido ${pixels.size}"
        }
    }

    operator fun get(x: Int, y: Int): Int = pixels[y * largura + x]
}

/**
 * Camada "nitidez": realça as bordas das barras (unsharp mask, como no Python).
 * Desfoque gaussiano 3x3 (pesos 1-2-1) calculado direto no vetor de pixels,
 * repetindo a borda da imagem nos cantos.
 */
fun aplicarNitidez(imagem: ImagemCinza): ImagemCinza {
    val largura = imagem.largura
    val altura = imagem.altura
    val p = imagem.pixels
    val saida = IntArray(p.size)
    for (y in 0 until altura) {
        val acima = maxOf(y - 1, 0) * largura
        val linha = y * largura
        val abaixo = minOf(y + 1, altura - 1) * largura
        for (x in 0 until largura) {
            val esquerda = maxOf(x - 1, 0)
            val direita = minOf(x + 1, largura - 1)
            val somaPonderada =
                p[acima + esquerda] + 2 * p[acima + x] + p[acima + direita] +
                    2 * p[linha + esquerda] + 4 * p[linha + x] + 2 * p[linha + direita] +
                    p[abaixo + esquerda] + 2 * p[abaixo + x] + p[abaixo + direita]
            val original = p[linha + x]
            val desfocado = somaPonderada / 16.0
            saida[linha + x] =
                (original + INTENSIDADE_NITIDEZ * (original - desfocado)).roundToInt().coerceIn(0, 255)
        }
    }
    return ImagemCinza(largura, altura, saida)
}

/**
 * Camada "contraste": estica os tons para usar toda a faixa 0..255.
 * Simplificação do CLAHE do Python, que exigiria OpenCV no Android.
 */
fun esticarContraste(imagem: ImagemCinza): ImagemCinza {
    val histograma = histograma(imagem)
    val corte = (imagem.pixels.size * PERCENTIL_CORTE_CONTRASTE).toInt()
    val minimo = tomNoAcumulado(histograma, corte)
    val maximo = tomNoAcumulado(histograma, imagem.pixels.size - 1 - corte)
    if (maximo <= minimo) return imagem

    val faixa = (maximo - minimo).toDouble()
    val saida = IntArray(imagem.pixels.size) { i ->
        ((imagem.pixels[i] - minimo) * 255 / faixa).roundToInt().coerceIn(0, 255)
    }
    return ImagemCinza(imagem.largura, imagem.altura, saida)
}

/** Camada "otsu": preto e branco puro, com limiar escolhido automaticamente. */
fun binarizarOtsu(imagem: ImagemCinza): ImagemCinza {
    val limiar = limiarOtsu(imagem)
    val saida = IntArray(imagem.pixels.size) { i -> if (imagem.pixels[i] > limiar) 255 else 0 }
    return ImagemCinza(imagem.largura, imagem.altura, saida)
}

fun limiarOtsu(imagem: ImagemCinza): Int {
    val histograma = histograma(imagem)
    val total = imagem.pixels.size.toDouble()
    val somaTotal = histograma.indices.sumOf { it.toDouble() * histograma[it] }

    var pesoFundo = 0.0
    var somaFundo = 0.0
    var melhorVariancia = -1.0
    var melhorLimiar = 0
    for (tom in 0..255) {
        pesoFundo += histograma[tom]
        if (pesoFundo == 0.0) continue
        val pesoFrente = total - pesoFundo
        if (pesoFrente == 0.0) break

        somaFundo += tom.toDouble() * histograma[tom]
        val mediaFundo = somaFundo / pesoFundo
        val mediaFrente = (somaTotal - somaFundo) / pesoFrente
        val varianciaEntreClasses = pesoFundo * pesoFrente * (mediaFundo - mediaFrente).pow(2)
        if (varianciaEntreClasses > melhorVariancia) {
            melhorVariancia = varianciaEntreClasses
            melhorLimiar = tom
        }
    }
    return melhorLimiar
}

private fun histograma(imagem: ImagemCinza): IntArray {
    val contagem = IntArray(256)
    imagem.pixels.forEach { contagem[it]++ }
    return contagem
}

/** Tom em que o histograma acumulado ultrapassa a posição pedida. */
private fun tomNoAcumulado(histograma: IntArray, posicao: Int): Int {
    var acumulado = 0
    for (tom in 0..255) {
        acumulado += histograma[tom]
        if (acumulado > posicao) return tom
    }
    return 255
}
