package br.com.interfocus.testemlkit

import org.junit.Assert.assertEquals
import org.junit.Test

class DetectorRfDetrTest {

    @Test
    fun converteCaixaNormalizadaParaPixelsEOrdenaPorConfianca() {
        val caixas = arrayOf(
            floatArrayOf(0.25f, 0.5f, 0.5f, 0.2f), // logit 0 -> confiança 0,50
            floatArrayOf(0.5f, 0.5f, 1.2f, 0.1f), // logit 3 -> 0,95; passa da borda
        )
        val logits = arrayOf(floatArrayOf(0f), floatArrayOf(3f))

        val resultado = selecionarDeteccoes(caixas, logits, larguraFoto = 1000, alturaFoto = 500)

        assertEquals(listOf(Caixa(0, 225, 1000, 275), Caixa(0, 200, 500, 300)), resultado)
    }

    @Test
    fun descartaConsultasAbaixoDaConfiancaMinima() {
        val caixas = arrayOf(floatArrayOf(0.5f, 0.5f, 0.1f, 0.1f))
        val logits = arrayOf(floatArrayOf(-5f)) // confiança ~0,007

        assertEquals(emptyList<Caixa>(), selecionarDeteccoes(caixas, logits, 100, 100))
    }

    @Test
    fun descartaCaixaForaDaFoto() {
        val caixas = arrayOf(floatArrayOf(1.3f, 0.5f, 0.2f, 0.2f)) // centro além da borda direita
        val logits = arrayOf(floatArrayOf(3f))

        assertEquals(emptyList<Caixa>(), selecionarDeteccoes(caixas, logits, 100, 100))
    }
}
