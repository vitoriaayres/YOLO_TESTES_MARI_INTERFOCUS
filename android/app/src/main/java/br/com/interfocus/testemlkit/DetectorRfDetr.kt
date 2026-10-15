package br.com.interfocus.testemlkit

import ai.onnxruntime.OnnxTensor
import ai.onnxruntime.OrtEnvironment
import ai.onnxruntime.OrtSession
import ai.onnxruntime.TensorInfo
import android.content.Context
import android.graphics.Bitmap
import java.nio.FloatBuffer
import kotlin.math.exp
import kotlin.math.roundToInt

private const val ARQUIVO_MODELO = "rfdetr_v2.onnx"
private const val CONFIANCA_MINIMA = 0.10f // mesmo corte do servidor (roboflow_min_confidence)
private const val MAXIMO_DETECCOES = 8 // como o localizador: mais caixas só gastariam o teto de tempo
private val MEDIA_IMAGENET = floatArrayOf(0.485f, 0.456f, 0.406f)
private val DESVIO_IMAGENET = floatArrayOf(0.229f, 0.224f, 0.225f)

/**
 * RF-DETR Small treinado para a classe "Barcode", rodando no aparelho com ONNX Runtime.
 * Substitui o envio da foto ao servidor: aponta as regiões que a 2ª passagem lê.
 * Pré e pós-processamento iguais aos do RFDETROnnxDetector do servidor (branch app_mobile).
 */
class DetectorRfDetr(private val context: Context) : AutoCloseable {

    private val ambiente = OrtEnvironment.getEnvironment()

    // Carregar o modelo leva alguns segundos: só acontece na primeira detecção, fora da thread de UI.
    private val sessaoPreguicosa = lazy {
        context.assets.open(ARQUIVO_MODELO).use { ambiente.createSession(it.readBytes()) }
    }

    /** Carrega o modelo antes da primeira foto, para ela não pagar esse tempo. Chamar fora da thread de UI. */
    fun preparar() {
        sessaoPreguicosa.value
    }

    /** Caixas dos códigos, na escala do [bitmap], da mais para a menos confiável. */
    fun detectar(bitmap: Bitmap): List<Caixa> {
        val sessao = sessaoPreguicosa.value
        val nomeEntrada = sessao.inputNames.first()
        val resolucao = (sessao.inputInfo.getValue(nomeEntrada).info as TensorInfo).shape.last().toInt()
        val forma = longArrayOf(1, 3, resolucao.toLong(), resolucao.toLong())

        OnnxTensor.createTensor(ambiente, paraTensor(bitmap, resolucao), forma).use { entrada ->
            sessao.run(mapOf(nomeEntrada to entrada)).use { saidas ->
                @Suppress("UNCHECKED_CAST")
                val caixas = (saidas[0].value as Array<Array<FloatArray>>)[0]
                @Suppress("UNCHECKED_CAST")
                val logits = (saidas[1].value as Array<Array<FloatArray>>)[0]
                return selecionarDeteccoes(caixas, logits, bitmap.width, bitmap.height)
            }
        }
    }

    override fun close() {
        if (sessaoPreguicosa.isInitialized()) sessaoPreguicosa.value.close()
    }

    /** Redimensiona para o quadrado do modelo e normaliza (ImageNet), em planos RGB (NCHW). */
    private fun paraTensor(bitmap: Bitmap, resolucao: Int): FloatBuffer {
        val reduzida = Bitmap.createScaledBitmap(bitmap, resolucao, resolucao, true)
        val pixels = IntArray(resolucao * resolucao)
        reduzida.getPixels(pixels, 0, resolucao, 0, 0, resolucao, resolucao)
        val plano = pixels.size
        val tensor = FloatBuffer.allocate(3 * plano)
        fun normalizar(valor: Int, canal: Int) = (valor / 255f - MEDIA_IMAGENET[canal]) / DESVIO_IMAGENET[canal]
        pixels.forEachIndexed { i, cor ->
            tensor.put(i, normalizar(cor shr 16 and 0xFF, 0))
            tensor.put(plano + i, normalizar(cor shr 8 and 0xFF, 1))
            tensor.put(2 * plano + i, normalizar(cor and 0xFF, 2))
        }
        return tensor
    }
}

/**
 * Converte a saída do RF-DETR em caixas: confiança = sigmoid do maior logit da consulta;
 * caixas em (centro x, centro y, largura, altura) normalizados para a foto.
 * O DETR não gera duplicatas, por isso não há NMS.
 */
fun selecionarDeteccoes(
    caixas: Array<FloatArray>,
    logits: Array<FloatArray>,
    larguraFoto: Int,
    alturaFoto: Int,
): List<Caixa> =
    logits.indices
        .map { consulta -> consulta to 1f / (1f + exp(-logits[consulta].max())) }
        .filter { (_, confianca) -> confianca >= CONFIANCA_MINIMA }
        .sortedByDescending { (_, confianca) -> confianca }
        .take(MAXIMO_DETECCOES)
        .map { (consulta, _) ->
            val (centroX, centroY, largura, altura) = caixas[consulta]
            Caixa(
                esquerda = ((centroX - largura / 2) * larguraFoto).roundToInt().coerceAtLeast(0),
                topo = ((centroY - altura / 2) * alturaFoto).roundToInt().coerceAtLeast(0),
                direita = ((centroX + largura / 2) * larguraFoto).roundToInt().coerceAtMost(larguraFoto),
                base = ((centroY + altura / 2) * alturaFoto).roundToInt().coerceAtMost(alturaFoto),
            )
        }
        // As caixas do RF-DETR não são limitadas a [0, 1]: uma caixa fora da foto vira largura 0
        // e derrubaria o Bitmap.createBitmap das camadas.
        .filter { it.largura > 1 && it.altura > 1 }
