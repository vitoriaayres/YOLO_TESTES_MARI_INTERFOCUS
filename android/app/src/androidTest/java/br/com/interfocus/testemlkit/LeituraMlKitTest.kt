package br.com.interfocus.testemlkit

import android.graphics.BitmapFactory
import android.util.Log
import androidx.test.ext.junit.runners.AndroidJUnit4
import androidx.test.platform.app.InstrumentationRegistry
import kotlinx.coroutines.runBlocking
import org.junit.After
import org.junit.Test
import org.junit.runner.RunWith
import java.io.File

private const val TAG = "MLKIT_TESTE"
private const val CABECALHO_CSV =
    "arquivo;so_imagem_inteira;valores_imagem_inteira;quantidade;valores;formatos;origens;votos;" +
        "candidatos;chamadas_mlkit;tempo_ms;tempo_mlkit_ms;erro"

/**
 * Roda o mesmo leitor do app (1ª + 2ª passagem) em todas as imagens de uma pasta
 * dos assets de teste. Argumentos opcionais (para comparar variações):
 *   -Pandroid.testInstrumentationRunnerArguments.pasta=dificeis
 *   -Pandroid.testInstrumentationRunnerArguments.estrategia=SO_GRADE
 *   -Pandroid.testInstrumentationRunnerArguments.detector=localizador
 *   -Pandroid.testInstrumentationRunnerArguments.orcamento=sem
 */
@RunWith(AndroidJUnit4::class)
class LeituraMlKitTest {

    private val instrumentacao = InstrumentationRegistry.getInstrumentation()
    private val argumentos = InstrumentationRegistry.getArguments()
    private val pastaImagens = argumentos.getString("pasta") ?: "imagens"
    private val estrategia = argumentos.getString("estrategia")
        ?.let(EstrategiaSegundaPassagem::valueOf)
        ?: EstrategiaSegundaPassagem.CANDIDATOS_E_GRADE
    // Candidatos do RF-DETR (padrão do app) ou, com detector=localizador, do localizador por gradiente.
    private val detector = if (argumentos.getString("detector") == "localizador") null
        else DetectorRfDetr(instrumentacao.targetContext)
    // Mesmo teto do app (a grade só roda no tempo que sobra); com orcamento=sem, mede tudo.
    private val orcamentoMs = if (argumentos.getString("orcamento") == "sem") Long.MAX_VALUE else ORCAMENTO_PADRAO_MS
    private val leitor = LeitorCodigos(estrategia, orcamentoMs, detector)

    @After
    fun liberarLeitor() {
        leitor.fechar()
        detector?.close()
    }

    @Test
    fun lerTodasAsImagens() {
        Log.i(TAG, "Pasta: $pastaImagens · estratégia: $estrategia")
        val nomes = instrumentacao.context.assets.list(pastaImagens).orEmpty().sorted()
        val linhas = mutableListOf(CABECALHO_CSV)

        nomes.forEach { nome -> linhas += lerImagem(nome) }

        salvarCsv(linhas)
        val lidas = linhas.drop(1).count { it.split(";")[3] != "0" }
        Log.i(TAG, "Concluído: $lidas de ${nomes.size} imagens com pelo menos um código de barras")
    }

    private fun lerImagem(nome: String): String {
        return try {
            val bitmap = instrumentacao.context.assets
                .open("$pastaImagens/$nome")
                .use(BitmapFactory::decodeStream)
                .let(::limitarTamanho)

            val resultado = runBlocking { leitor.ler(bitmap) }
            val codigos = resultado.codigos

            val linha = listOf(
                nome,
                resultado.codigosImagemInteira.size,
                resultado.codigosImagemInteira.joinToString("|") { it.valor },
                codigos.size,
                codigos.joinToString("|") { it.valor },
                codigos.joinToString("|") { it.formato.nome },
                codigos.joinToString("|") { it.origem },
                codigos.joinToString("|") { it.votos.toString() },
                resultado.regioesCandidatas,
                resultado.chamadasMlKit,
                resultado.tempoMs,
                resultado.tempoMlKitMs,
                "",
            ).joinToString(";")
            Log.i(TAG, linha)
            linha
        } catch (erro: Exception) {
            Log.e(TAG, "Falha em $nome", erro)
            "$nome;0;;0;;;;;0;0;0;0;${erro.javaClass.simpleName}"
        }
    }

    private fun salvarCsv(linhas: List<String>) {
        val pasta = instrumentacao.targetContext.getExternalFilesDir(null)
        // O detector entra no nome para a rodada com o localizador não sobrescrever a do RF-DETR.
        val nomeDetector = if (detector == null) "localizador" else "rfdetr"
        val arquivo = File(pasta, "resultado_${pastaImagens}_${estrategia}_$nomeDetector.csv")
        arquivo.writeText(linhas.joinToString("\n"))
        Log.i(TAG, "CSV salvo em ${arquivo.absolutePath}")
    }
}
