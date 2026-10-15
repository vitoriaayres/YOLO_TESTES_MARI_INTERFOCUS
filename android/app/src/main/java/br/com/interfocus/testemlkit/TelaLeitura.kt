package br.com.interfocus.testemlkit

import android.graphics.Bitmap
import android.graphics.Paint
import android.graphics.Typeface
import androidx.activity.compose.rememberLauncherForActivityResult
import androidx.activity.result.PickVisualMediaRequest
import androidx.activity.result.contract.ActivityResultContracts.PickVisualMedia
import androidx.compose.foundation.Canvas
import androidx.compose.foundation.Image
import androidx.compose.foundation.background
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.aspectRatio
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.navigationBarsPadding
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.size
import androidx.compose.foundation.layout.width
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.shape.CircleShape
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.foundation.text.selection.SelectionContainer
import androidx.compose.foundation.verticalScroll
import androidx.compose.material3.Button
import androidx.compose.material3.Card
import androidx.compose.material3.CardDefaults
import androidx.compose.material3.CircularProgressIndicator
import androidx.compose.material3.Icon
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.OutlinedButton
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.DisposableEffect
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.rememberCoroutineScope
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clip
import androidx.compose.ui.geometry.CornerRadius
import androidx.compose.ui.geometry.Offset
import androidx.compose.ui.geometry.Size
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.graphics.asImageBitmap
import androidx.compose.ui.graphics.drawscope.DrawScope
import androidx.compose.ui.graphics.drawscope.Stroke
import androidx.compose.ui.graphics.nativeCanvas
import androidx.compose.ui.graphics.toArgb
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.res.painterResource
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.text.style.TextAlign
import androidx.compose.ui.unit.dp
import kotlinx.coroutines.CancellationException
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext
import java.util.Locale

private sealed interface EstadoLeitura {
    data object Vazio : EstadoLeitura
    data object Lendo : EstadoLeitura
    data class Sucesso(val resultado: ResultadoLeitura) : EstadoLeitura
    data class Erro(val mensagem: String) : EstadoLeitura
}

@Composable
fun TelaLeitura(detector: DetectorRfDetr, modifier: Modifier = Modifier) {
    val context = LocalContext.current
    val escopo = rememberCoroutineScope()
    val leitor = remember(detector) { LeitorCodigos(detector = detector) }
    DisposableEffect(leitor) { onDispose { leitor.fechar() } }

    var imagem by remember { mutableStateOf<Bitmap?>(null) }
    var estado by remember { mutableStateOf<EstadoLeitura>(EstadoLeitura.Vazio) }

    val seletorDeFoto = rememberLauncherForActivityResult(PickVisualMedia()) { uri ->
        if (uri == null) return@rememberLauncherForActivityResult
        escopo.launch {
            estado = EstadoLeitura.Lendo
            estado = try {
                val bitmap = withContext(Dispatchers.IO) { carregarBitmap(context, uri) }
                imagem = bitmap
                EstadoLeitura.Sucesso(leitor.ler(bitmap))
            } catch (erro: CancellationException) {
                throw erro
            } catch (erro: Exception) {
                EstadoLeitura.Erro(erro.message ?: erro.javaClass.simpleName)
            }
        }
    }
    val escolherFoto = { seletorDeFoto.launch(PickVisualMediaRequest(PickVisualMedia.ImageOnly)) }

    Column(
        modifier = modifier
            .fillMaxSize()
            .background(MaterialTheme.colorScheme.background)
            .verticalScroll(rememberScrollState())
            .navigationBarsPadding()
            .padding(16.dp),
        verticalArrangement = Arrangement.spacedBy(16.dp),
    ) {
        val atual = estado
        if (atual == EstadoLeitura.Vazio) {
            EstadoInicial(aoEscolher = escolherFoto)
            return@Column
        }

        val codigos = (atual as? EstadoLeitura.Sucesso)?.resultado?.codigos.orEmpty()
        imagem?.let { CartaoImagem(it, codigos) }
        when (atual) {
            EstadoLeitura.Lendo -> CartaoAnalisando()
            is EstadoLeitura.Erro -> CartaoErro(atual.mensagem)
            is EstadoLeitura.Sucesso -> {
                ResumoLeitura(atual.resultado)
                ListaDeCodigos(atual.resultado.codigos)
            }
            EstadoLeitura.Vazio -> Unit
        }
        OutlinedButton(
            onClick = escolherFoto,
            enabled = atual != EstadoLeitura.Lendo,
            modifier = Modifier.fillMaxWidth(),
        ) { Text("Escolher outra foto") }
    }
}

@Composable
private fun EstadoInicial(aoEscolher: () -> Unit) {
    Card(
        colors = CardDefaults.cardColors(containerColor = MaterialTheme.colorScheme.surface),
        elevation = CardDefaults.cardElevation(defaultElevation = 1.dp),
        modifier = Modifier.fillMaxWidth(),
    ) {
        Column(
            horizontalAlignment = Alignment.CenterHorizontally,
            modifier = Modifier
                .fillMaxWidth()
                .padding(28.dp),
        ) {
            Box(
                contentAlignment = Alignment.Center,
                modifier = Modifier
                    .size(72.dp)
                    .background(MaterialTheme.colorScheme.primaryContainer, CircleShape),
            ) {
                Icon(
                    painterResource(R.drawable.ic_imagem),
                    contentDescription = null,
                    tint = MaterialTheme.colorScheme.primary,
                    modifier = Modifier.size(32.dp),
                )
            }
            Spacer(Modifier.size(20.dp))
            Text("Analisar uma foto", style = MaterialTheme.typography.titleMedium, fontWeight = FontWeight.Bold)
            Spacer(Modifier.size(8.dp))
            Text(
                "Use quando a câmera não conseguir ler. A foto passa pela leitura completa: " +
                    "imagem inteira e regiões suspeitas com 5 camadas de tratamento cada.",
                style = MaterialTheme.typography.bodyMedium,
                color = MaterialTheme.colorScheme.onSurfaceVariant,
                textAlign = TextAlign.Center,
            )
            Spacer(Modifier.size(24.dp))
            Button(onClick = aoEscolher, modifier = Modifier.fillMaxWidth()) { Text("Escolher foto") }
        }
    }
}

@Composable
private fun CartaoImagem(bitmap: Bitmap, codigos: List<CodigoLido>) {
    val corDestaque = MaterialTheme.colorScheme.primary
    val corNumero = MaterialTheme.colorScheme.onPrimary
    val pincelNumero = remember {
        Paint().apply {
            isAntiAlias = true
            textAlign = Paint.Align.CENTER
            typeface = Typeface.DEFAULT_BOLD
        }
    }
    Box(
        modifier = Modifier
            .fillMaxWidth()
            .aspectRatio(bitmap.width.toFloat() / bitmap.height)
            .clip(RoundedCornerShape(16.dp)),
    ) {
        Image(
            bitmap = bitmap.asImageBitmap(),
            contentDescription = "Foto analisada",
            modifier = Modifier.fillMaxSize(),
        )
        Canvas(Modifier.fillMaxSize()) {
            val escala = size.width / bitmap.width
            codigos.forEachIndexed { indice, codigo ->
                val caixa = codigo.caixa ?: return@forEachIndexed
                val topoEsquerdo = Offset(caixa.esquerda * escala, caixa.topo * escala)
                val tamanho = Size(caixa.largura * escala, caixa.altura * escala)
                val raio = CornerRadius(4.dp.toPx())
                drawRoundRect(corDestaque.copy(alpha = 0.15f), topoEsquerdo, tamanho, raio)
                drawRoundRect(corDestaque, topoEsquerdo, tamanho, raio, Stroke(2.5.dp.toPx()))
                desenharNumero(indice + 1, topoEsquerdo, corDestaque, corNumero, pincelNumero)
            }
        }
    }
}

/** Círculo numerado no canto da caixa, igual ao número da lista abaixo da foto. */
private fun DrawScope.desenharNumero(numero: Int, canto: Offset, fundo: Color, texto: Color, pincel: Paint) {
    val raio = 11.dp.toPx()
    val centro = Offset(canto.x, canto.y).let {
        Offset(it.x.coerceAtLeast(raio), it.y.coerceAtLeast(raio))
    }
    drawCircle(fundo, raio, centro)
    pincel.color = texto.toArgb()
    pincel.textSize = 13.dp.toPx()
    val baseTexto = centro.y - (pincel.descent() + pincel.ascent()) / 2
    drawContext.canvas.nativeCanvas.drawText(numero.toString(), centro.x, baseTexto, pincel)
}

@Composable
private fun CartaoAnalisando() {
    Card(modifier = Modifier.fillMaxWidth()) {
        Row(verticalAlignment = Alignment.CenterVertically, modifier = Modifier.padding(20.dp)) {
            CircularProgressIndicator(Modifier.size(28.dp), strokeWidth = 3.dp)
            Spacer(Modifier.width(16.dp))
            Column {
                Text("Analisando a foto…", style = MaterialTheme.typography.titleSmall, fontWeight = FontWeight.SemiBold)
                Text(
                    "Imagem inteira + regiões × 5 camadas (até ${ORCAMENTO_PADRAO_MS / 1000} s)",
                    style = MaterialTheme.typography.bodySmall,
                    color = MaterialTheme.colorScheme.onSurfaceVariant,
                )
            }
        }
    }
}

@Composable
private fun CartaoErro(mensagem: String) {
    Card(
        colors = CardDefaults.cardColors(containerColor = MaterialTheme.colorScheme.errorContainer),
        modifier = Modifier.fillMaxWidth(),
    ) {
        Column(Modifier.padding(20.dp)) {
            Text(
                "Não foi possível analisar a foto",
                style = MaterialTheme.typography.titleSmall,
                fontWeight = FontWeight.SemiBold,
                color = MaterialTheme.colorScheme.onErrorContainer,
            )
            Text(mensagem, style = MaterialTheme.typography.bodySmall, color = MaterialTheme.colorScheme.onErrorContainer)
        }
    }
}

@Composable
private fun ResumoLeitura(resultado: ResultadoLeitura) {
    val segundos = String.format(Locale("pt", "BR"), "%.1f s", resultado.tempoMs / 1000.0)
    val ganhoCamadas = resultado.codigos.size - resultado.codigosImagemInteira.size
    Column(verticalArrangement = Arrangement.spacedBy(8.dp)) {
        Row(horizontalArrangement = Arrangement.spacedBy(8.dp)) {
            Indicador("Códigos", resultado.codigos.size.toString(), Modifier.weight(1f))
            Indicador("Tempo", segundos, Modifier.weight(1f))
            Indicador("Leituras ML Kit", resultado.chamadasMlKit.toString(), Modifier.weight(1f))
        }
        if (resultado.interrompidaPorTempo) {
            Text(
                "A análise parou no limite de ${ORCAMENTO_PADRAO_MS / 1000} s; algumas regiões não foram verificadas.",
                style = MaterialTheme.typography.bodySmall,
                color = MaterialTheme.colorScheme.onSurfaceVariant,
            )
        }
        if (ganhoCamadas > 0) {
            Text(
                "As 5 camadas encontraram $ganhoCamadas código(s) a mais que a leitura simples.",
                style = MaterialTheme.typography.bodySmall,
                color = MaterialTheme.colorScheme.primary,
            )
        }
    }
}

@Composable
private fun Indicador(rotulo: String, valor: String, modifier: Modifier = Modifier) {
    Card(
        colors = CardDefaults.cardColors(containerColor = MaterialTheme.colorScheme.surface),
        elevation = CardDefaults.cardElevation(defaultElevation = 1.dp),
        modifier = modifier,
    ) {
        Column(Modifier.padding(horizontal = 12.dp, vertical = 10.dp)) {
            Text(valor, style = MaterialTheme.typography.titleLarge, fontWeight = FontWeight.Bold)
            Text(rotulo, style = MaterialTheme.typography.labelSmall, color = MaterialTheme.colorScheme.onSurfaceVariant)
        }
    }
}

@Composable
private fun ListaDeCodigos(codigos: List<CodigoLido>) {
    if (codigos.isEmpty()) {
        Text(
            "Nenhum código de barras encontrado nesta foto. Tente uma foto mais próxima e com o código nítido.",
            style = MaterialTheme.typography.bodyMedium,
            color = MaterialTheme.colorScheme.onSurfaceVariant,
        )
        return
    }
    SelectionContainer {
        Column(verticalArrangement = Arrangement.spacedBy(8.dp)) {
            codigos.forEachIndexed { indice, codigo -> CartaoCodigo(indice + 1, codigo) }
        }
    }
}

@Composable
private fun CartaoCodigo(numero: Int, codigo: CodigoLido) {
    Card(
        colors = CardDefaults.cardColors(containerColor = MaterialTheme.colorScheme.surface),
        elevation = CardDefaults.cardElevation(defaultElevation = 1.dp),
        modifier = Modifier.fillMaxWidth(),
    ) {
        Row(verticalAlignment = Alignment.CenterVertically, modifier = Modifier.padding(14.dp)) {
            MarcadorNumero(numero)
            Spacer(Modifier.width(12.dp))
            Column(Modifier.weight(1f), verticalArrangement = Arrangement.spacedBy(4.dp)) {
                ValorCodigo(codigo.valor)
                Row(verticalAlignment = Alignment.CenterVertically) {
                    EtiquetaFormato(codigo.formato)
                    Spacer(Modifier.width(8.dp))
                    Text(
                        "${codigo.origem} · ${codigo.votos} voto(s)",
                        style = MaterialTheme.typography.labelSmall,
                        color = MaterialTheme.colorScheme.onSurfaceVariant,
                        maxLines = 1,
                    )
                }
            }
        }
    }
}
