package br.com.interfocus.testemlkit

import org.junit.Assert.assertEquals
import org.junit.Test

class GeometriaBlocosTest {

    @Test
    fun gradeTemNoveBlocosDeMetadeDoTamanho() {
        val blocos = gerarBlocos(1000, 800)

        assertEquals(9, blocos.size)
        blocos.forEach {
            assertEquals(500, it.largura)
            assertEquals(400, it.altura)
        }
    }

    @Test
    fun gradeCobreDoCantoSuperiorAoInferior() {
        val blocos = gerarBlocos(1000, 800)

        assertEquals(Caixa(0, 0, 500, 400), blocos.first())
        assertEquals(Caixa(250, 200, 750, 600), blocos[4]) // bloco central
        assertEquals(Caixa(500, 400, 1000, 800), blocos.last())
    }

    @Test
    fun transformacaoVoltaParaCoordenadasDaFotoOriginal() {
        // Bloco começando em (100, 50), com margem de 10 px e ampliado 2x.
        val transformacao = Transformacao(origemX = 100, origemY = 50, margem = 10, escala = 2f)

        val original = transformacao.paraOriginal(Caixa(40, 20, 240, 120))

        assertEquals(Caixa(110, 50, 210, 100), original)
    }

    @Test
    fun expandirCaixaLevaParaEscalaDaFotoComMargem() {
        // 200x20: margem = 20 * 0,5 + 200 * 0,05 = 20 px na imagem reduzida, depois x2.
        val expandida = expandirCaixa(Caixa(100, 50, 300, 70), escala = 2f, larguraFoto = 2000, alturaFoto = 2000)
        assertEquals(Caixa(160, 60, 640, 180), expandida)
    }

    @Test
    fun expandirCaixaNaoPassaDaBordaDaFoto() {
        val expandida = expandirCaixa(Caixa(0, 0, 100, 10), escala = 2f, larguraFoto = 210, alturaFoto = 30)
        assertEquals(Caixa(0, 0, 210, 30), expandida)
    }

    @Test
    fun transformacaoSemAmpliacaoSoDesfazMargemEDeslocamento() {
        val transformacao = Transformacao(origemX = 300, origemY = 0, margem = 20, escala = 1f)

        assertEquals(Caixa(300, 0, 400, 30), transformacao.paraOriginal(Caixa(20, 20, 120, 50)))
    }
}
