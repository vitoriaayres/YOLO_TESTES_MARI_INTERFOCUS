package br.com.interfocus.testemlkit

/*
 * Regras de confirmação portadas do pipeline Python (barcode_pipeline.py):
 *   - ONE_DIMENSIONAL_BARCODE_TYPES -> FormatoCodigo (somente 1D)
 *   - keep_confirmed_barcodes       -> filtrarCodigosConfirmados
 *       * descarta leitura sem valor
 *       * prioriza Code 128/39/93 sobre EAN/UPC (um EAN dentro de um Code 128
 *         costuma ser fragmento dele)
 *       * remove duplicatas por IoU >= 0,20 ou contenção >= 0,70
 *       * devolve em ordem de leitura (de cima para baixo, da esquerda para a direita)
 *   - choose_consensus_decode_for_band -> consolidarLeituras
 *       * agrupa leituras iguais vindas de camadas/blocos diferentes (votos)
 *       * EAN/UPC só é aceito com pelo menos 2 votos, porque um fragmento de
 *         Code 128 pode ser lido como EAN por engano
 *
 * Este arquivo não depende do Android nem do ML Kit, para poder ser testado
 * com JUnit puro.
 */

const val ORIGEM_IMAGEM_INTEIRA = "imagem inteira"

private const val IOU_DUPLICATA = 0.20 // confirmed_nms_iou
internal const val CONTENCAO_DUPLICATA = 0.70 // confirmed_containment
private const val VOTOS_MINIMOS_VAREJO = 2

/**
 * [verificadorObrigatorio]: o formato tem dígito verificador forte e obrigatório
 * (Code 128 e Code 93), então uma leitura errada é praticamente impossível e
 * um único voto basta. Nos demais, o verificador é opcional ou fraco.
 */
enum class FormatoCodigo(val nome: String, val alfanumerico: Boolean, val verificadorObrigatorio: Boolean) {
    CODE_128("Code 128", alfanumerico = true, verificadorObrigatorio = true),
    CODE_39("Code 39", alfanumerico = true, verificadorObrigatorio = false),
    CODE_93("Code 93", alfanumerico = true, verificadorObrigatorio = true),
    CODABAR("Codabar", alfanumerico = false, verificadorObrigatorio = false),
    ITF("ITF", alfanumerico = false, verificadorObrigatorio = false),
    EAN_13("EAN-13", alfanumerico = false, verificadorObrigatorio = false),
    EAN_8("EAN-8", alfanumerico = false, verificadorObrigatorio = false),
    UPC_A("UPC-A", alfanumerico = false, verificadorObrigatorio = false),
    UPC_E("UPC-E", alfanumerico = false, verificadorObrigatorio = false),
}

data class Caixa(val esquerda: Int, val topo: Int, val direita: Int, val base: Int) {
    val largura: Int get() = maxOf(0, direita - esquerda)
    val altura: Int get() = maxOf(0, base - topo)
    val area: Long get() = largura.toLong() * altura

    /** Usado na câmera ao vivo: só vale o código cujo centro está dentro da moldura. */
    fun contemCentroDe(outra: Caixa): Boolean {
        val centroX = (outra.esquerda + outra.direita) / 2
        val centroY = (outra.topo + outra.base) / 2
        return centroX in esquerda..direita && centroY in topo..base
    }
}

data class CodigoLido(
    val valor: String,
    val formato: FormatoCodigo,
    val caixa: Caixa?,
    val origem: String = ORIGEM_IMAGEM_INTEIRA, // bloco e camada que leram o código
    val votos: Int = 1, // quantas perspectivas diferentes leram o mesmo valor
)

fun consolidarLeituras(leituras: List<CodigoLido>): List<CodigoLido> =
    leituras
        .filter { it.valor.isNotBlank() }
        .groupBy { it.formato to it.valor }
        .values
        .map { grupo ->
            val principal = grupo.firstOrNull { it.caixa != null } ?: grupo.first()
            principal.copy(votos = grupo.map { it.origem }.distinct().size)
        }
        .filter { it.formato.alfanumerico || it.votos >= VOTOS_MINIMOS_VAREJO }

/**
 * Parada antecipada da segunda passagem (fast_mode do Python): o bloco já está
 * resolvido quando leu algo e cada valor lido é confiável — formato com
 * verificador obrigatório, ou pelo menos 2 camadas concordando.
 */
fun blocoResolvido(leiturasDoBloco: List<CodigoLido>): Boolean =
    leiturasDoBloco.isNotEmpty() &&
        leiturasDoBloco.groupBy { it.formato to it.valor }.all { (chave, grupo) ->
            chave.first.verificadorObrigatorio || grupo.map { it.origem }.distinct().size >= 2
        }

fun filtrarCodigosConfirmados(codigos: List<CodigoLido>): List<CodigoLido> {
    val porPrioridade = codigos
        .filter { it.valor.isNotBlank() }
        .sortedWith(
            compareByDescending<CodigoLido> { it.formato.alfanumerico }.thenByDescending { it.votos }
        )

    val selecionados = mutableListOf<CodigoLido>()
    for (codigo in porPrioridade) {
        if (selecionados.none { ehMesmaRegiao(codigo, it) }) selecionados += codigo
    }
    return selecionados.sortedWith(
        compareBy({ it.caixa?.topo ?: Int.MAX_VALUE }, { it.caixa?.esquerda ?: Int.MAX_VALUE })
    )
}

private fun ehMesmaRegiao(a: CodigoLido, b: CodigoLido): Boolean {
    val caixaA = a.caixa ?: return false
    val caixaB = b.caixa ?: return false
    return intersecaoSobreUniao(caixaA, caixaB) >= IOU_DUPLICATA ||
        intersecaoSobreMenor(caixaA, caixaB) >= CONTENCAO_DUPLICATA
}

private fun areaIntersecao(a: Caixa, b: Caixa): Long {
    val largura = minOf(a.direita, b.direita) - maxOf(a.esquerda, b.esquerda)
    val altura = minOf(a.base, b.base) - maxOf(a.topo, b.topo)
    return if (largura <= 0 || altura <= 0) 0L else largura.toLong() * altura
}

private fun intersecaoSobreUniao(a: Caixa, b: Caixa): Double {
    val intersecao = areaIntersecao(a, b)
    val uniao = a.area + b.area - intersecao
    return if (uniao > 0) intersecao.toDouble() / uniao else 0.0
}

internal fun intersecaoSobreMenor(a: Caixa, b: Caixa): Double {
    val menor = minOf(a.area, b.area)
    return if (menor > 0) areaIntersecao(a, b).toDouble() / menor else 0.0
}
