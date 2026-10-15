package br.com.interfocus.testemlkit

/*
 * Consenso entre frames da câmera ao vivo. É o equivalente, no tempo real, ao
 * consenso por votos entre camadas do pipeline Python: cada frame é uma
 * "perspectiva" diferente do código (ângulo, foco e luz variam um pouco), e um
 * valor só é confirmado depois de aparecer em N dos últimos [janela] frames.
 * Isso evita mandar para a API uma leitura errada de um frame só.
 *
 * N depende do formato: Code 128 e Code 93 têm dígito verificador obrigatório,
 * então 2 frames bastam; os demais exigem 3.
 */

private const val FRAMES_COM_VERIFICADOR = 2
private const val FRAMES_SEM_VERIFICADOR = 3
private const val JANELA_FRAMES = 5

fun framesParaConfirmar(formato: FormatoCodigo): Int =
    if (formato.verificadorObrigatorio) FRAMES_COM_VERIFICADOR else FRAMES_SEM_VERIFICADOR

class ConsensoFrames(
    private val framesNecessarios: (FormatoCodigo) -> Int = ::framesParaConfirmar,
    private val janela: Int = JANELA_FRAMES,
) {
    /** Mesmo número de frames para todos os formatos. */
    constructor(framesNecessarios: Int, janela: Int = JANELA_FRAMES) : this({ framesNecessarios }, janela) {
        require(framesNecessarios in 1..janela) { "framesNecessarios deve estar entre 1 e $janela" }
    }

    private val historico = ArrayDeque<Set<Pair<FormatoCodigo, String>>>()
    private val jaConfirmados = mutableSetOf<Pair<FormatoCodigo, String>>()

    /** Registra os códigos lidos num frame e devolve só os que acabaram de ser confirmados. */
    fun registrar(codigosDoFrame: List<CodigoLido>): List<CodigoLido> {
        historico.addLast(codigosDoFrame.map(::chave).toSet())
        if (historico.size > janela) historico.removeFirst()

        // Código que saiu da janela pode ser confirmado de novo quando voltar.
        val presentesNaJanela = historico.flatten().toSet()
        jaConfirmados.retainAll(presentesNaJanela)

        val novos = codigosDoFrame.filter { codigo ->
            val chave = chave(codigo)
            val necessarios = framesNecessarios(codigo.formato).coerceIn(1, janela)
            chave !in jaConfirmados && historico.count { chave in it } >= necessarios
        }
        jaConfirmados += novos.map(::chave)
        return novos
    }

    fun limpar() {
        historico.clear()
        jaConfirmados.clear()
    }

    private fun chave(codigo: CodigoLido) = codigo.formato to codigo.valor
}
