package br.com.interfocus.testemlkit

import android.graphics.Bitmap
import android.graphics.Canvas
import android.graphics.Color
import kotlin.math.roundToInt

/*
 * As 5 camadas aplicadas a cada bloco, na ordem do barcode_layer_pipeline do Python.
 * Cada camada é uma nova "perspectiva" do mesmo trecho para o ML Kit:
 *   1. recorte   -> bloco com borda branca (zona de silêncio ao redor do código)
 *   2. ampliado  -> super-resolução (até 4x)
 *   3. nitidez   -> realce das bordas das barras
 *   4. contraste -> esticamento de tons (no lugar do CLAHE)
 *   5. otsu      -> preto e branco puro
 * A etapa "grayscale" do Python não vira camada própria porque o ML Kit já
 * converte tudo para tons de cinza internamente; ela acontece dentro da camada 3.
 */

private const val MARGEM_RELATIVA = 0.04f
private const val ESCALA_MAXIMA = 4f // super_resolution_max_scale
private const val LADO_MAXIMO_AMPLIADO = 1600 // limite para não estourar a memória

class CamadaRecorte(val nome: String, val imagem: Bitmap, val transformacao: Transformacao)

fun gerarCamadas(imagem: Bitmap, bloco: Caixa): Sequence<CamadaRecorte> = sequence {
    val margem = (maxOf(bloco.largura, bloco.altura) * MARGEM_RELATIVA).roundToInt()
    val trecho = Bitmap.createBitmap(imagem, bloco.esquerda, bloco.topo, bloco.largura, bloco.altura)
    val recorte = comMargemBranca(trecho, margem)
    yield(CamadaRecorte("recorte", recorte, Transformacao(bloco.esquerda, bloco.topo, margem, 1f)))

    val ampliado = ampliar(recorte)
    val escalaReal = ampliado.width.toFloat() / recorte.width
    val transformacaoAmpliada = Transformacao(bloco.esquerda, bloco.topo, margem, escalaReal)
    yield(CamadaRecorte("ampliado", ampliado, transformacaoAmpliada))

    val nitidez = aplicarNitidez(paraCinza(ampliado))
    yield(CamadaRecorte("nitidez", paraBitmap(nitidez), transformacaoAmpliada))

    val contraste = esticarContraste(nitidez)
    yield(CamadaRecorte("contraste", paraBitmap(contraste), transformacaoAmpliada))

    yield(CamadaRecorte("otsu", paraBitmap(binarizarOtsu(contraste)), transformacaoAmpliada))
}

private fun comMargemBranca(bitmap: Bitmap, margem: Int): Bitmap {
    val saida = Bitmap.createBitmap(
        bitmap.width + 2 * margem,
        bitmap.height + 2 * margem,
        Bitmap.Config.ARGB_8888,
    )
    Canvas(saida).apply {
        drawColor(Color.WHITE)
        drawBitmap(bitmap, margem.toFloat(), margem.toFloat(), null)
    }
    return saida
}

private fun ampliar(bitmap: Bitmap): Bitmap {
    val maiorLado = maxOf(bitmap.width, bitmap.height)
    val escala = (LADO_MAXIMO_AMPLIADO.toFloat() / maiorLado).coerceIn(1f, ESCALA_MAXIMA)
    return Bitmap.createScaledBitmap(
        bitmap,
        (bitmap.width * escala).roundToInt(),
        (bitmap.height * escala).roundToInt(),
        true,
    )
}

internal fun paraCinza(bitmap: Bitmap): ImagemCinza {
    val cores = IntArray(bitmap.width * bitmap.height)
    bitmap.getPixels(cores, 0, bitmap.width, 0, 0, bitmap.width, bitmap.height)
    val tons = IntArray(cores.size) { i ->
        val cor = cores[i]
        (Color.red(cor) * 299 + Color.green(cor) * 587 + Color.blue(cor) * 114) / 1000
    }
    return ImagemCinza(bitmap.width, bitmap.height, tons)
}

private fun paraBitmap(imagem: ImagemCinza): Bitmap {
    val cores = IntArray(imagem.pixels.size) { i ->
        val tom = imagem.pixels[i]
        Color.rgb(tom, tom, tom)
    }
    return Bitmap.createBitmap(cores, imagem.largura, imagem.altura, Bitmap.Config.ARGB_8888)
}
