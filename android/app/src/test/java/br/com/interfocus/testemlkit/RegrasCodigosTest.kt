package br.com.interfocus.testemlkit

import org.junit.Assert.assertEquals
import org.junit.Test

class RegrasCodigosTest {

    private fun codigo(valor: String, formato: FormatoCodigo, caixa: Caixa?) =
        CodigoLido(valor, formato, caixa)

    @Test
    fun descartaLeituraSemValor() {
        val resultado = filtrarCodigosConfirmados(
            listOf(codigo("  ", FormatoCodigo.CODE_128, Caixa(0, 0, 100, 30)))
        )
        assertEquals(emptyList<CodigoLido>(), resultado)
    }

    @Test
    fun eanDentroDeCode128EhDescartadoComoFragmento() {
        val code128 = codigo("ZTEG12345678", FormatoCodigo.CODE_128, Caixa(0, 0, 400, 60))
        val fragmento = codigo("1234567890128", FormatoCodigo.EAN_13, Caixa(50, 5, 250, 55))

        val resultado = filtrarCodigosConfirmados(listOf(fragmento, code128))

        assertEquals(listOf(code128), resultado)
    }

    @Test
    fun codigosEmRegioesSeparadasSaoMantidos() {
        val mac = codigo("0A1B2C3D4E5F", FormatoCodigo.CODE_128, Caixa(0, 0, 300, 50))
        val serial = codigo("ZTEG00000001", FormatoCodigo.CODE_128, Caixa(0, 100, 300, 150))

        val resultado = filtrarCodigosConfirmados(listOf(serial, mac))

        assertEquals(listOf(mac, serial), resultado)
    }

    @Test
    fun resultadoSaiEmOrdemDeLeitura() {
        val baixo = codigo("C", FormatoCodigo.CODE_128, Caixa(0, 200, 100, 230))
        val topoDireita = codigo("B", FormatoCodigo.CODE_128, Caixa(300, 0, 400, 30))
        val topoEsquerda = codigo("A", FormatoCodigo.CODE_39, Caixa(0, 0, 100, 30))

        val resultado = filtrarCodigosConfirmados(listOf(baixo, topoDireita, topoEsquerda))

        assertEquals(listOf("A", "B", "C"), resultado.map { it.valor })
    }

    @Test
    fun consolidacaoContaVotosDeOrigensDiferentes() {
        val caixa = Caixa(0, 0, 300, 50)
        val leituras = listOf(
            CodigoLido("ZTEG1", FormatoCodigo.CODE_128, caixa, origem = "bloco 1 · recorte"),
            CodigoLido("ZTEG1", FormatoCodigo.CODE_128, caixa, origem = "bloco 1 · otsu"),
            CodigoLido("ZTEG1", FormatoCodigo.CODE_128, caixa, origem = "bloco 1 · otsu"), // repetida
        )

        val resultado = consolidarLeituras(leituras)

        assertEquals(1, resultado.size)
        assertEquals(2, resultado.single().votos)
        assertEquals("bloco 1 · recorte", resultado.single().origem)
    }

    @Test
    fun eanComUmVotoSoEhDescartado() {
        val ean = CodigoLido("7891234567895", FormatoCodigo.EAN_13, Caixa(0, 0, 200, 60))

        assertEquals(emptyList<CodigoLido>(), consolidarLeituras(listOf(ean)))
    }

    @Test
    fun eanComDoisVotosEhAceito() {
        val caixa = Caixa(0, 0, 200, 60)
        val leituras = listOf(
            CodigoLido("7891234567895", FormatoCodigo.EAN_13, caixa, origem = "bloco 2 · ampliado"),
            CodigoLido("7891234567895", FormatoCodigo.EAN_13, caixa, origem = "bloco 2 · contraste"),
        )

        assertEquals(2, consolidarLeituras(leituras).single().votos)
    }

    @Test
    fun naMesmaRegiaoVenceOValorComMaisVotos() {
        val caixa = Caixa(0, 0, 300, 50)
        val maisVotado = CodigoLido("ZTEGAAA", FormatoCodigo.CODE_128, caixa, votos = 4)
        val leituraErrada = CodigoLido("ZTEGAAB", FormatoCodigo.CODE_128, caixa, votos = 1)

        assertEquals(listOf(maisVotado), filtrarCodigosConfirmados(listOf(leituraErrada, maisVotado)))
    }

    @Test
    fun blocoSemLeituraNaoEstaResolvido() {
        assertEquals(false, blocoResolvido(emptyList()))
    }

    @Test
    fun blocoComCode128EstaResolvidoComUmVoto() {
        val leitura = CodigoLido("ZTEG1", FormatoCodigo.CODE_128, null, origem = "bloco 1 · recorte")
        assertEquals(true, blocoResolvido(listOf(leitura)))
    }

    @Test
    fun blocoComEanPrecisaDeDuasCamadasConcordando() {
        val recorte = CodigoLido("7891234567895", FormatoCodigo.EAN_13, null, origem = "bloco 1 · recorte")
        val ampliado = recorte.copy(origem = "bloco 1 · ampliado")

        assertEquals(false, blocoResolvido(listOf(recorte)))
        assertEquals(true, blocoResolvido(listOf(recorte, ampliado)))
    }

    @Test
    fun codigoSemCaixaNaoEhTratadoComoDuplicata() {
        val semCaixa = codigo("X1", FormatoCodigo.CODE_128, null)
        val comCaixa = codigo("X2", FormatoCodigo.CODE_128, Caixa(0, 0, 100, 30))

        val resultado = filtrarCodigosConfirmados(listOf(semCaixa, comCaixa))

        assertEquals(listOf(comCaixa, semCaixa), resultado)
    }
}
