package br.com.interfocus.testemlkit

import android.content.Context
import android.graphics.Bitmap
import android.graphics.ImageDecoder
import android.net.Uri
import android.os.SystemClock
import android.util.Log
import com.google.mlkit.vision.barcode.BarcodeScanner
import com.google.mlkit.vision.barcode.BarcodeScannerOptions
import com.google.mlkit.vision.barcode.BarcodeScanning
import com.google.mlkit.vision.barcode.ZoomSuggestionOptions
import com.google.mlkit.vision.barcode.common.Barcode
import com.google.mlkit.vision.common.InputImage
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.suspendCancellableCoroutine
import kotlinx.coroutines.withContext
import kotlinx.coroutines.withTimeoutOrNull
import kotlin.coroutines.resume
import kotlin.coroutines.resumeWithException
import kotlin.math.roundToInt

private const val LADO_MAXIMO_PX = 2048 // evita falta de memória com fotos muito grandes

/** Formato do ML Kit -> formato 1D aceito pelo app. QR, Data Matrix etc. ficam de fora. */
private val FORMATOS_ML_KIT = mapOf(
    Barcode.FORMAT_CODE_128 to FormatoCodigo.CODE_128,
    Barcode.FORMAT_CODE_39 to FormatoCodigo.CODE_39,
    Barcode.FORMAT_CODE_93 to FormatoCodigo.CODE_93,
    Barcode.FORMAT_CODABAR to FormatoCodigo.CODABAR,
    Barcode.FORMAT_ITF to FormatoCodigo.ITF,
    Barcode.FORMAT_EAN_13 to FormatoCodigo.EAN_13,
    Barcode.FORMAT_EAN_8 to FormatoCodigo.EAN_8,
    Barcode.FORMAT_UPC_A to FormatoCodigo.UPC_A,
    Barcode.FORMAT_UPC_E to FormatoCodigo.UPC_E,
)

private const val LADO_DETECCAO_PX = 800 // o localizador roda numa cópia reduzida, por velocidade

// Teto da leitura por foto: em aparelho lento (ou emulador sem memória) as até ~86 chamadas
// do ML Kit podiam levar dezenas de minutos. Estourado o teto, devolve o que já foi lido.
const val ORCAMENTO_PADRAO_MS = 10_000L
private const val TEMPO_MAXIMO_CHAMADA_MS = 5_000L // protege contra uma chamada do ML Kit que nunca responde
private const val TAG_LEITURA = "LeitorCodigos" // adb logcat -s LeitorCodigos: onde o tempo de cada foto foi gasto

data class ResultadoLeitura(
    val codigos: List<CodigoLido>,
    val codigosImagemInteira: List<CodigoLido>, // só a 1ª passagem, para comparar
    val regioesCandidatas: Int,
    val chamadasMlKit: Int,
    val tempoMs: Long,
    val tempoMlKitMs: Long, // o restante do tempo é gasto localizando e gerando as camadas
    val interrompidaPorTempo: Boolean = false, // a 2ª passagem parou no teto de tempo
)

/** Onde a 2ª passagem procura: regiões apontadas pelo localizador, a grade fixa ou as duas. */
enum class EstrategiaSegundaPassagem(val usarCandidatos: Boolean, val usarGrade: Boolean) {
    SO_GRADE(usarCandidatos = false, usarGrade = true),
    SO_CANDIDATOS(usarCandidatos = true, usarGrade = false),
    CANDIDATOS_E_GRADE(usarCandidatos = true, usarGrade = true),
}

/**
 * Leitura em duas passagens, espelhando o pipeline Python:
 *   1ª: ML Kit na imagem inteira (equivale à "leitura global rápida").
 *   2ª: regiões passam pelas 5 camadas, nesta ordem e até o teto de tempo: caixas do
 *       RF-DETR, regiões do localizador de barras que ele não cobriu, grade fixa de blocos.
 *       Todas as leituras viram votos e passam pelas mesmas regras de confirmação.
 */
class LeitorCodigos(
    private val estrategia: EstrategiaSegundaPassagem = EstrategiaSegundaPassagem.CANDIDATOS_E_GRADE,
    private val orcamentoMs: Long = ORCAMENTO_PADRAO_MS, // a avaliação passa Long.MAX_VALUE para medir tudo
    private val detector: DetectorRfDetr? = null, // sem ele, os candidatos vêm do localizador por gradiente
) {

    private val scanner = criarScanner1D()

    suspend fun ler(bitmap: Bitmap): ResultadoLeitura = withContext(Dispatchers.Default) {
        val inicio = SystemClock.elapsedRealtime()
        var tempoMlKit = 0L
        var chamadas = 0
        var interrompida = false

        fun estourouTempo() = (SystemClock.elapsedRealtime() - inicio >= orcamentoMs).also { if (it) interrompida = true }

        suspend fun detectarMedindo(imagem: Bitmap, origem: String, paraOriginal: (Caixa) -> Caixa): List<CodigoLido> {
            val inicioChamada = SystemClock.elapsedRealtime()
            val codigos = withTimeoutOrNull(TEMPO_MAXIMO_CHAMADA_MS) { detectar(imagem, origem, paraOriginal) }
            tempoMlKit += SystemClock.elapsedRealtime() - inicioChamada
            chamadas++
            return codigos.orEmpty()
        }

        /** Passa uma região pelas 5 camadas, parando assim que a leitura for confiável. */
        suspend fun lerRegiao(regiao: Caixa, nome: String): List<CodigoLido> {
            val leiturasDaRegiao = mutableListOf<CodigoLido>()
            // A sequência é preguiçosa: ao parar, as camadas seguintes nem são geradas.
            for (camada in gerarCamadas(bitmap, regiao)) {
                if (estourouTempo()) break
                val origem = "$nome · ${camada.nome}"
                leiturasDaRegiao += detectarMedindo(camada.imagem, origem, camada.transformacao::paraOriginal)
                if (blocoResolvido(leiturasDaRegiao)) break
            }
            return leiturasDaRegiao
        }

        val primeiraPassagem = detectarMedindo(bitmap, ORIGEM_IMAGEM_INTEIRA) { it }
        val leituras = primeiraPassagem.toMutableList()

        // Candidatos antes da grade: são as regiões mais prováveis, então ficam com o tempo garantido.
        val inicioCandidatos = SystemClock.elapsedRealtime()
        val candidatos = if (estrategia.usarCandidatos) regioesCandidatas(bitmap) else emptyList()
        val tempoCandidatos = SystemClock.elapsedRealtime() - inicioCandidatos
        candidatos.forEachIndexed { indice, regiao ->
            if (!estourouTempo()) leituras += lerRegiao(regiao, "candidato ${indice + 1}")
        }
        // A grade vem por último: só roda no tempo que sobrar depois dos candidatos.
        if (estrategia.usarGrade) {
            gerarBlocos(bitmap.width, bitmap.height).forEachIndexed { indice, bloco ->
                if (!estourouTempo()) leituras += lerRegiao(bloco, "bloco ${indice + 1}")
            }
        }

        val tempoTotal = SystemClock.elapsedRealtime() - inicio
        Log.i(
            TAG_LEITURA,
            "total=${tempoTotal}ms candidatos=${candidatos.size} em ${tempoCandidatos}ms " +
                "mlkit=${tempoMlKit}ms em $chamadas chamadas, camadas=${tempoTotal - tempoCandidatos - tempoMlKit}ms " +
                "interrompida=$interrompida",
        )
        ResultadoLeitura(
            codigos = confirmar(leituras),
            codigosImagemInteira = confirmar(primeiraPassagem),
            regioesCandidatas = candidatos.size,
            chamadasMlKit = chamadas,
            tempoMs = tempoTotal,
            tempoMlKitMs = tempoMlKit,
            interrompidaPorTempo = interrompida,
        )
    }

    fun fechar() = scanner.close()

    /**
     * Regiões suspeitas na escala da foto: primeiro as do RF-DETR; depois as do localizador
     * que o RF-DETR não cobriu (nas fotos difíceis, o localizador ainda acha códigos que o modelo não marca).
     */
    private fun regioesCandidatas(bitmap: Bitmap): List<Caixa> {
        val doLocalizador = regioesDoLocalizador(bitmap)
        val doDetector = detector?.detectar(bitmap)
            ?.map { expandirCaixa(it, 1f, bitmap.width, bitmap.height) }
            ?: return doLocalizador
        val naoCobertas = doLocalizador.filter { regiao ->
            doDetector.none { intersecaoSobreMenor(it, regiao) >= CONTENCAO_DUPLICATA }
        }
        return doDetector + naoCobertas
    }

    /** Roda o localizador numa cópia reduzida e devolve as regiões já na escala da foto. */
    private fun regioesDoLocalizador(bitmap: Bitmap): List<Caixa> {
        val fator = maxOf(bitmap.width, bitmap.height).toFloat() / LADO_DETECCAO_PX
        val reduzida = if (fator <= 1f) bitmap else Bitmap.createScaledBitmap(
            bitmap,
            (bitmap.width / fator).roundToInt(),
            (bitmap.height / fator).roundToInt(),
            true,
        )
        val escala = bitmap.width.toFloat() / reduzida.width
        return localizarCandidatos(paraCinza(reduzida))
            .map { expandirCaixa(it, escala, bitmap.width, bitmap.height) }
            .filter { it.largura > 1 && it.altura > 1 }
    }

    private fun confirmar(leituras: List<CodigoLido>) =
        filtrarCodigosConfirmados(consolidarLeituras(leituras))

    private suspend fun detectar(
        imagem: Bitmap,
        origem: String,
        paraOriginal: (Caixa) -> Caixa,
    ): List<CodigoLido> = suspendCancellableCoroutine { continuacao ->
        scanner.process(InputImage.fromBitmap(imagem, 0))
            .addOnSuccessListener { codigos ->
                continuacao.resume(codigos.mapNotNull { converterCodigo(it, origem, paraOriginal) })
            }
            .addOnFailureListener(continuacao::resumeWithException)
    }
}

/**
 * Scanner do ML Kit restrito aos formatos 1D. Usado pela foto e pela câmera ao vivo.
 * [zoomAutomatico] só faz sentido na câmera: o ML Kit pede zoom quando o código está longe.
 */
fun criarScanner1D(zoomAutomatico: ZoomSuggestionOptions? = null): BarcodeScanner {
    val opcoes = BarcodeScannerOptions.Builder()
        .setBarcodeFormats(FORMATOS_ML_KIT.keys.first(), *FORMATOS_ML_KIT.keys.drop(1).toIntArray())
    zoomAutomatico?.let(opcoes::setZoomSuggestionOptions)
    return BarcodeScanning.getClient(opcoes.build())
}

/** Converte o resultado do ML Kit; devolve null para formatos fora da lista 1D. */
fun converterCodigo(
    codigo: Barcode,
    origem: String,
    paraOriginal: (Caixa) -> Caixa = { it },
): CodigoLido? {
    val formato = FORMATOS_ML_KIT[codigo.format] ?: return null
    val caixa = codigo.boundingBox?.let { paraOriginal(Caixa(it.left, it.top, it.right, it.bottom)) }
    return CodigoLido(codigo.rawValue.orEmpty(), formato, caixa, origem)
}

/**
 * Carrega a imagem já na orientação correta (respeita o EXIF da foto) e
 * em bitmap de software, que é o que o ML Kit consegue ler.
 */
fun carregarBitmap(context: Context, uri: Uri): Bitmap {
    val fonte = ImageDecoder.createSource(context.contentResolver, uri)
    return ImageDecoder.decodeBitmap(fonte) { decoder, info, _ ->
        decoder.allocator = ImageDecoder.ALLOCATOR_SOFTWARE
        val maiorLado = maxOf(info.size.width, info.size.height)
        if (maiorLado > LADO_MAXIMO_PX) {
            val escala = LADO_MAXIMO_PX.toFloat() / maiorLado
            decoder.setTargetSize(
                (info.size.width * escala).toInt(),
                (info.size.height * escala).toInt(),
            )
        }
    }
}

/** Mesmo limite de tamanho do carregarBitmap, para imagens que já estão em memória. */
fun limitarTamanho(bitmap: Bitmap): Bitmap {
    val maiorLado = maxOf(bitmap.width, bitmap.height)
    if (maiorLado <= LADO_MAXIMO_PX) return bitmap
    val escala = LADO_MAXIMO_PX.toFloat() / maiorLado
    return Bitmap.createScaledBitmap(
        bitmap,
        (bitmap.width * escala).roundToInt(),
        (bitmap.height * escala).roundToInt(),
        true,
    )
}
