package br.com.interfocus.testemlkit

import kotlinx.coroutines.delay

/**
 * Consulta se um código de barras existe no sistema.
 * Contrato previsto da API real: POST /leituras {valor, formato, origem} -> {existe}.
 */
interface VerificadorCodigos {
    suspend fun existe(codigo: CodigoLido): Boolean
}

/**
 * API simulada enquanto o endpoint real não existe. Para trocar pela real, basta
 * criar outra implementação de [VerificadorCodigos] que faça a chamada HTTP.
 */
class VerificadorSimulado(
    private val codigosCadastrados: Set<String> = CODIGOS_DE_EXEMPLO,
    private val latenciaMs: Long = 300, // imita o tempo de rede
) : VerificadorCodigos {

    override suspend fun existe(codigo: CodigoLido): Boolean {
        delay(latenciaMs)
        return codigo.valor in codigosCadastrados
    }
}

/** Valores fictícios no formato dos seriais reais, para a simulação responder "existe". */
private val CODIGOS_DE_EXEMPLO = setOf(
    "0A1B2C3D4E5F",
    "ZTEG00000001",
    "001122334455",
    "ZTEG00000002",
    "KQDL00000001",
    "ZTEG00000003",
    "ZTEPQT1R0000001",
)
