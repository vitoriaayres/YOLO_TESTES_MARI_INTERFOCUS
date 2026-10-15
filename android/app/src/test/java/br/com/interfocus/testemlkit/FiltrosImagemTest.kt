package br.com.interfocus.testemlkit

import org.junit.Assert.assertArrayEquals
import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Test

class FiltrosImagemTest {

    /** Imagem 10x10: metade esquerda com tom [escuro], metade direita com tom [claro]. */
    private fun imagemDuasMetades(escuro: Int, claro: Int): ImagemCinza {
        val pixels = IntArray(100) { i -> if (i % 10 < 5) escuro else claro }
        return ImagemCinza(10, 10, pixels)
    }

    @Test
    fun limiarOtsuFicaEntreOsDoisTons() {
        val limiar = limiarOtsu(imagemDuasMetades(20, 220))
        assertTrue("limiar $limiar fora do intervalo", limiar in 20 until 220)
    }

    @Test
    fun otsuGeraSomentePretoEBranco() {
        val resultado = binarizarOtsu(imagemDuasMetades(90, 160))
        assertEquals(setOf(0, 255), resultado.pixels.toSet())
        assertEquals(0, resultado[0, 0])
        assertEquals(255, resultado[9, 0])
    }

    @Test
    fun contrasteEsticaTonsParaFaixaCompleta() {
        val resultado = esticarContraste(imagemDuasMetades(100, 150))
        assertEquals(0, resultado[0, 0])
        assertEquals(255, resultado[9, 0])
    }

    @Test
    fun contrasteNaoAlteraImagemDeTomUnico() {
        val uniforme = ImagemCinza(4, 4, IntArray(16) { 128 })
        assertArrayEquals(uniforme.pixels, esticarContraste(uniforme).pixels)
    }

    @Test
    fun nitidezNaoAlteraAreaUniforme() {
        val uniforme = ImagemCinza(5, 5, IntArray(25) { 77 })
        assertArrayEquals(uniforme.pixels, aplicarNitidez(uniforme).pixels)
    }

    @Test
    fun nitidezAumentaDiferencaNaBorda() {
        val resultado = aplicarNitidez(imagemDuasMetades(100, 200))
        assertTrue(resultado[4, 5] < 100) // lado escuro da borda fica mais escuro
        assertTrue(resultado[5, 5] > 200) // lado claro da borda fica mais claro
    }
}
