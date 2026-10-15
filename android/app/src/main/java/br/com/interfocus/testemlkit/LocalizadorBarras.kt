package br.com.interfocus.testemlkit

/*
 * Localizador de regiões com cara de código de barras, em Kotlin puro.
 *
 * Faz o papel que o detector do Roboflow tinha no pipeline Python: apontar ONDE
 * estão os códigos, para que cada um seja recortado justo e ampliado
 * (super-resolução) antes da leitura. Sem rede neural e sem custo.
 *
 * Princípio: um código de barras tem muita variação de claro/escuro na direção
 * perpendicular às barras e quase nenhuma na direção delas. Texto e bordas têm
 * variação nas duas direções, e fundo liso não tem nenhuma.
 *
 * Passos (para cada orientação, deitado e em pé):
 *   1. resposta = |gradiente perpendicular às barras| - |gradiente ao longo delas|
 *   2. média numa janela (suaviza e junta as barras)
 *   3. limiar pelo percentil (as regiões mais "barradas" da foto)
 *   4. fechamento morfológico alongado (une barras vizinhas num bloco só)
 *   5. componentes conectados com tamanho e proporção de código de barras
 */

private const val RAIO_SUAVIZACAO = 4 // janela 9x9
private const val PERCENTIL_LIMIAR = 0.90
private const val RESPOSTA_MINIMA = 12
private const val RAIO_FECHAMENTO_LONGO = 10 // ao longo do código: une as barras
private const val RAIO_FECHAMENTO_CURTO = 2
private const val AREA_MINIMA_RELATIVA = 0.0008
private const val COMPRIMENTO_MINIMO_RELATIVO = 0.04
private const val PROPORCAO_MINIMA = 1.5
private const val MAXIMO_CANDIDATOS = 8

private enum class Orientacao { DEITADO, EM_PE } // deitado = barras em pé, código na horizontal

fun localizarCandidatos(imagem: ImagemCinza): List<Caixa> =
    Orientacao.entries
        .flatMap { candidatosNaOrientacao(imagem, it) }
        .sortedByDescending { it.area }
        .take(MAXIMO_CANDIDATOS)

private fun candidatosNaOrientacao(imagem: ImagemCinza, orientacao: Orientacao): List<Caixa> {
    val largura = imagem.largura
    val altura = imagem.altura
    val suavizada = mediaNaJanela(respostaDeBarras(imagem, orientacao), largura, altura, RAIO_SUAVIZACAO)
    val limiar = maxOf(percentil(suavizada, PERCENTIL_LIMIAR), RESPOSTA_MINIMA)
    val mascara = BooleanArray(suavizada.size) { suavizada[it] > limiar }

    val (raioX, raioY) = when (orientacao) {
        Orientacao.DEITADO -> RAIO_FECHAMENTO_LONGO to RAIO_FECHAMENTO_CURTO
        Orientacao.EM_PE -> RAIO_FECHAMENTO_CURTO to RAIO_FECHAMENTO_LONGO
    }
    val fechada = erodir(dilatar(mascara, largura, altura, raioX, raioY), largura, altura, raioX, raioY)
    return componentesConectados(fechada, largura, altura).filter { pareceCodigo(it, largura, altura, orientacao) }
}

/** Gradiente perpendicular às barras menos o gradiente ao longo delas (nunca negativo). */
private fun respostaDeBarras(imagem: ImagemCinza, orientacao: Orientacao): IntArray {
    val largura = imagem.largura
    val altura = imagem.altura
    val p = imagem.pixels
    val resposta = IntArray(p.size)
    for (y in 1 until altura - 1) {
        for (x in 1 until largura - 1) {
            val i = y * largura + x
            val gradienteX = kotlin.math.abs(p[i + 1] - p[i - 1])
            val gradienteY = kotlin.math.abs(p[i + largura] - p[i - largura])
            val diferenca = when (orientacao) {
                Orientacao.DEITADO -> gradienteX - gradienteY
                Orientacao.EM_PE -> gradienteY - gradienteX
            }
            resposta[i] = maxOf(diferenca, 0)
        }
    }
    return resposta
}

private fun pareceCodigo(caixa: Caixa, largura: Int, altura: Int, orientacao: Orientacao): Boolean {
    val (comprimento, espessura, ladoImagem) = when (orientacao) {
        Orientacao.DEITADO -> Triple(caixa.largura, caixa.altura, largura)
        Orientacao.EM_PE -> Triple(caixa.altura, caixa.largura, altura)
    }
    return caixa.area >= AREA_MINIMA_RELATIVA * largura * altura &&
        comprimento >= COMPRIMENTO_MINIMO_RELATIVO * ladoImagem &&
        comprimento >= PROPORCAO_MINIMA * espessura
}

// ------------------------------------------------------------ operações de imagem

/** Média de cada pixel numa janela quadrada, via imagem integral (custo independe do raio). */
private fun mediaNaJanela(valores: IntArray, largura: Int, altura: Int, raio: Int): IntArray {
    val integral = imagemIntegral(largura, altura) { valores[it].toLong() }
    return IntArray(valores.size) { i ->
        val x = i % largura
        val y = i / largura
        val janela = janela(x, y, largura, altura, raio, raio)
        (somaNaJanela(integral, largura, janela) / janela.area).toInt()
    }
}

private fun dilatar(mascara: BooleanArray, largura: Int, altura: Int, raioX: Int, raioY: Int): BooleanArray {
    val integral = imagemIntegral(largura, altura) { if (mascara[it]) 1L else 0L }
    return BooleanArray(mascara.size) { i ->
        somaNaJanela(integral, largura, janela(i % largura, i / largura, largura, altura, raioX, raioY)) > 0
    }
}

private fun erodir(mascara: BooleanArray, largura: Int, altura: Int, raioX: Int, raioY: Int): BooleanArray {
    val integral = imagemIntegral(largura, altura) { if (mascara[it]) 1L else 0L }
    return BooleanArray(mascara.size) { i ->
        val janela = janela(i % largura, i / largura, largura, altura, raioX, raioY)
        somaNaJanela(integral, largura, janela) == janela.area
    }
}

/** Janela em volta de (x, y), cortada nas bordas da imagem. */
private fun janela(x: Int, y: Int, largura: Int, altura: Int, raioX: Int, raioY: Int) = Caixa(
    esquerda = maxOf(x - raioX, 0),
    topo = maxOf(y - raioY, 0),
    direita = minOf(x + raioX + 1, largura),
    base = minOf(y + raioY + 1, altura),
)

/** integral[(y)*(largura+1) + x] = soma dos valores no retângulo [0, x) x [0, y). */
private inline fun imagemIntegral(largura: Int, altura: Int, valor: (Int) -> Long): LongArray {
    val passo = largura + 1
    val integral = LongArray(passo * (altura + 1))
    for (y in 0 until altura) {
        var somaDaLinha = 0L
        for (x in 0 until largura) {
            somaDaLinha += valor(y * largura + x)
            integral[(y + 1) * passo + (x + 1)] = integral[y * passo + (x + 1)] + somaDaLinha
        }
    }
    return integral
}

private fun somaNaJanela(integral: LongArray, largura: Int, janela: Caixa): Long {
    val passo = largura + 1
    return integral[janela.base * passo + janela.direita] - integral[janela.topo * passo + janela.direita] -
        integral[janela.base * passo + janela.esquerda] + integral[janela.topo * passo + janela.esquerda]
}

private fun percentil(valores: IntArray, fracao: Double): Int {
    val maximo = valores.maxOrNull() ?: return 0
    val contagem = IntArray(maximo + 1)
    valores.forEach { contagem[it]++ }
    val alvo = (valores.size * fracao).toLong()
    var acumulado = 0L
    for (valor in contagem.indices) {
        acumulado += contagem[valor]
        if (acumulado > alvo) return valor
    }
    return maximo
}

/** Caixas que envolvem cada grupo de pixels marcados (vizinhança de 4). */
private fun componentesConectados(mascara: BooleanArray, largura: Int, altura: Int): List<Caixa> {
    val visitado = BooleanArray(mascara.size)
    val fila = IntArray(mascara.size)
    val caixas = mutableListOf<Caixa>()
    for (inicio in mascara.indices) {
        if (!mascara[inicio] || visitado[inicio]) continue
        var esquerda = largura
        var topo = altura
        var direita = 0
        var base = 0
        var cabeca = 0
        var cauda = 0
        fila[cauda++] = inicio
        visitado[inicio] = true
        while (cabeca < cauda) {
            val i = fila[cabeca++]
            val x = i % largura
            val y = i / largura
            esquerda = minOf(esquerda, x)
            topo = minOf(topo, y)
            direita = maxOf(direita, x + 1)
            base = maxOf(base, y + 1)
            for (vizinho in intArrayOf(i - 1, i + 1, i - largura, i + largura)) {
                val mesmaLinha = vizinho / largura == y
                val vizinhoValido = vizinho in mascara.indices &&
                    (vizinho == i - largura || vizinho == i + largura || mesmaLinha)
                if (vizinhoValido && mascara[vizinho] && !visitado[vizinho]) {
                    visitado[vizinho] = true
                    fila[cauda++] = vizinho
                }
            }
        }
        caixas += Caixa(esquerda, topo, direita, base)
    }
    return caixas
}
