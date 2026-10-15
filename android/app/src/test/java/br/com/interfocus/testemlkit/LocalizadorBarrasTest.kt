package br.com.interfocus.testemlkit

import org.junit.Assert.assertTrue
import org.junit.Test

class LocalizadorBarrasTest {

    private val largura = 400
    private val altura = 300

    private fun imagemBranca() = ImagemCinza(largura, altura, IntArray(largura * altura) { 255 })

    /** Desenha barras pretas de 3 px alternadas dentro da [regiao]. */
    private fun ImagemCinza.comBarras(regiao: Caixa, codigoDeitado: Boolean): ImagemCinza {
        for (y in regiao.topo until regiao.base) {
            for (x in regiao.esquerda until regiao.direita) {
                val posicaoAoLongo = if (codigoDeitado) x else y
                if ((posicaoAoLongo / 3) % 2 == 0) pixels[y * largura + x] = 0
            }
        }
        return this
    }

    private fun List<Caixa>.algumaCobre(regiao: Caixa) =
        any { it.contemCentroDe(regiao) && regiao.contemCentroDe(it) }

    @Test
    fun encontraCodigoDeitado() {
        val regiao = Caixa(100, 130, 300, 170)
        val candidatos = localizarCandidatos(imagemBranca().comBarras(regiao, codigoDeitado = true))
        assertTrue("candidatos: $candidatos", candidatos.algumaCobre(regiao))
    }

    @Test
    fun encontraCodigoEmPe() {
        val regiao = Caixa(180, 50, 220, 250)
        val candidatos = localizarCandidatos(imagemBranca().comBarras(regiao, codigoDeitado = false))
        assertTrue("candidatos: $candidatos", candidatos.algumaCobre(regiao))
    }

    @Test
    fun encontraDoisCodigosEmpilhados() {
        val mac = Caixa(80, 80, 320, 110)
        val serial = Caixa(80, 160, 320, 190)
        val imagem = imagemBranca().comBarras(mac, codigoDeitado = true).comBarras(serial, codigoDeitado = true)
        val candidatos = localizarCandidatos(imagem)
        assertTrue("candidatos: $candidatos", candidatos.algumaCobre(mac) && candidatos.algumaCobre(serial))
    }

    @Test
    fun imagemLisaNaoTemCandidatos() {
        assertTrue(localizarCandidatos(imagemBranca()).isEmpty())
    }
}
