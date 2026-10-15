package br.com.interfocus.testemlkit

import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertTrue
import org.junit.Test

class ConsensoFramesTest {

    private val serial = CodigoLido("ZTEG12345678", FormatoCodigo.CODE_128, Caixa(0, 0, 300, 50))
    private val mac = CodigoLido("0A1B2C3D4E5F", FormatoCodigo.CODE_128, Caixa(0, 100, 300, 150))

    @Test
    fun soConfirmaNoTerceiroFrame() {
        val consenso = ConsensoFrames(framesNecessarios = 3, janela = 5)

        assertTrue(consenso.registrar(listOf(serial)).isEmpty())
        assertTrue(consenso.registrar(listOf(serial)).isEmpty())
        assertEquals(listOf(serial), consenso.registrar(listOf(serial)))
    }

    @Test
    fun toleraFrameSemLeituraNoMeio() {
        val consenso = ConsensoFrames(framesNecessarios = 3, janela = 5)

        consenso.registrar(listOf(serial))
        consenso.registrar(emptyList()) // foco oscilou neste frame
        consenso.registrar(listOf(serial))

        assertEquals(listOf(serial), consenso.registrar(listOf(serial)))
    }

    @Test
    fun naoConfirmaDuasVezesEnquantoContinuaNaTela() {
        val consenso = ConsensoFrames(framesNecessarios = 2, janela = 3)

        consenso.registrar(listOf(serial))
        assertEquals(listOf(serial), consenso.registrar(listOf(serial)))
        assertTrue(consenso.registrar(listOf(serial)).isEmpty())
    }

    @Test
    fun leituraIsoladaNuncaEhConfirmada() {
        val consenso = ConsensoFrames(framesNecessarios = 3, janela = 5)
        val leituraErrada = serial.copy(valor = "ZTEG12345679")

        val confirmados = listOf(
            consenso.registrar(listOf(serial)),
            consenso.registrar(listOf(leituraErrada)),
            consenso.registrar(listOf(serial)),
            consenso.registrar(listOf(serial)),
        ).flatten()

        assertFalse(leituraErrada in confirmados)
        assertTrue(serial in confirmados)
    }

    @Test
    fun code128ConfirmaEmDoisFramesEEanEmTres() {
        val consenso = ConsensoFrames() // regra padrão por formato
        val ean = CodigoLido("7891234567895", FormatoCodigo.EAN_13, Caixa(0, 200, 200, 260))

        assertTrue(consenso.registrar(listOf(serial, ean)).isEmpty())
        assertEquals(listOf(serial), consenso.registrar(listOf(serial, ean)))
        assertEquals(listOf(ean), consenso.registrar(listOf(serial, ean)))
    }

    @Test
    fun confirmaCodigosDiferentesDeFormaIndependente() {
        val consenso = ConsensoFrames(framesNecessarios = 2, janela = 3)

        consenso.registrar(listOf(serial, mac))
        assertEquals(listOf(serial, mac), consenso.registrar(listOf(serial, mac)))
    }
}
