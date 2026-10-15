package br.com.interfocus.testemlkit

import android.Manifest
import android.content.pm.PackageManager
import android.graphics.Bitmap
import android.graphics.Matrix
import android.os.SystemClock
import android.util.Size
import androidx.camera.core.ImageCapture
import androidx.camera.core.ImageCaptureException
import androidx.camera.core.ImageProxy
import androidx.compose.foundation.clickable
import androidx.compose.runtime.mutableLongStateOf
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.asExecutor
import kotlinx.coroutines.delay
import kotlinx.coroutines.suspendCancellableCoroutine
import kotlin.coroutines.resume
import kotlin.coroutines.resumeWithException
import androidx.activity.compose.rememberLauncherForActivityResult
import androidx.activity.result.contract.ActivityResultContracts.RequestPermission
import androidx.camera.core.ImageAnalysis.COORDINATE_SYSTEM_VIEW_REFERENCED
import androidx.camera.core.resolutionselector.ResolutionSelector
import androidx.camera.core.resolutionselector.ResolutionStrategy
import androidx.camera.mlkit.vision.MlKitAnalyzer
import androidx.camera.view.LifecycleCameraController
import androidx.camera.view.PreviewView
import androidx.compose.animation.core.RepeatMode
import androidx.compose.animation.core.animateFloat
import androidx.compose.animation.core.infiniteRepeatable
import androidx.compose.animation.core.rememberInfiniteTransition
import androidx.compose.animation.core.tween
import androidx.compose.foundation.Canvas
import androidx.compose.foundation.background
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.heightIn
import androidx.compose.foundation.layout.navigationBarsPadding
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.size
import androidx.compose.foundation.layout.width
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.shape.CircleShape
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.foundation.verticalScroll
import androidx.compose.material3.Button
import androidx.compose.material3.CircularProgressIndicator
import androidx.compose.material3.HorizontalDivider
import androidx.compose.material3.Icon
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Surface
import androidx.compose.material3.Text
import androidx.compose.material3.TextButton
import androidx.compose.runtime.Composable
import androidx.compose.runtime.DisposableEffect
import androidx.compose.runtime.LaunchedEffect
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateListOf
import androidx.compose.runtime.mutableStateMapOf
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.rememberCoroutineScope
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.geometry.CornerRadius
import androidx.compose.ui.geometry.Offset
import androidx.compose.ui.geometry.Rect
import androidx.compose.ui.geometry.RoundRect
import androidx.compose.ui.graphics.ClipOp
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.graphics.Path
import androidx.compose.ui.graphics.StrokeCap
import androidx.compose.ui.graphics.drawscope.DrawScope
import androidx.compose.ui.graphics.drawscope.Stroke
import androidx.compose.ui.graphics.drawscope.clipPath
import androidx.compose.ui.hapticfeedback.HapticFeedbackType
import androidx.compose.ui.layout.onSizeChanged
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.platform.LocalHapticFeedback
import androidx.compose.ui.res.painterResource
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.text.style.TextAlign
import androidx.compose.ui.unit.IntSize
import androidx.compose.ui.unit.dp
import androidx.compose.ui.viewinterop.AndroidView
import androidx.core.content.ContextCompat
import androidx.lifecycle.compose.LocalLifecycleOwner
import br.com.interfocus.testemlkit.ui.theme.LaranjaAlerta
import br.com.interfocus.testemlkit.ui.theme.VerdeSucesso
import br.com.interfocus.testemlkit.ui.theme.VerdeSucessoClaro
import br.com.interfocus.testemlkit.ui.theme.VermelhoErro
import com.google.mlkit.vision.barcode.BarcodeScanner
import com.google.mlkit.vision.barcode.ZoomSuggestionOptions
import kotlinx.coroutines.CancellationException
import kotlinx.coroutines.launch
import androidx.compose.ui.geometry.Size as TamanhoCompose

private const val ORIGEM_CAMERA = "câmera ao vivo"
private const val MAXIMO_CODIGOS_NA_LISTA = 6
private val RESOLUCAO_ANALISE = Size(1920, 1080) // códigos finos precisam de resolução
private const val ZOOM_MAXIMO_AUTOMATICO = 4f // além disso a imagem treme demais na mão

// Moldura de mira: faixa larga no centro da tela, no formato de código de barras.
// Larga o suficiente para códigos longos (seriais GPON) caberem inteiros.
private const val MIRA_MARGEM_LATERAL = 0.04f
private const val MIRA_ALTURA = 0.40f
private val ALTURA_MAXIMA_LISTA = 168.dp // ~3 leituras; o resto rola, para sobrar tela para a câmera
private const val OPACIDADE_MASCARA = 0.55f
private const val TEMPO_ATE_SUGERIR_FOTO_MS = 5_000L
private const val ORIGEM_FOTO = "foto · RF-DETR"

private enum class SituacaoCodigo(val texto: String, val cor: Color) {
    CONSULTANDO("consultando", Color.Gray),
    EXISTE("EXISTE", VerdeSucesso),
    NAO_ENCONTRADO("NÃO ENCONTRADO", VermelhoErro),
    ERRO("erro na consulta", LaranjaAlerta),
}

/** Situação do fallback: foto analisada no aparelho (RF-DETR + 5 camadas). */
private sealed interface AnaliseFoto {
    data object Ocioso : AnaliseFoto
    data object Analisando : AnaliseFoto
    data object SemCodigos : AnaliseFoto
    data class Falhou(val mensagem: String) : AnaliseFoto
}

@Composable
fun TelaCamera(
    verificador: VerificadorCodigos,
    detector: DetectorRfDetr,
    modifier: Modifier = Modifier,
) {
    val context = LocalContext.current
    var temPermissao by remember {
        mutableStateOf(
            ContextCompat.checkSelfPermission(context, Manifest.permission.CAMERA) ==
                PackageManager.PERMISSION_GRANTED
        )
    }
    val pedirPermissao = rememberLauncherForActivityResult(RequestPermission()) { temPermissao = it }
    LaunchedEffect(Unit) {
        if (!temPermissao) pedirPermissao.launch(Manifest.permission.CAMERA)
    }

    if (temPermissao) {
        CameraComLeitura(verificador, detector, modifier)
    } else {
        PedidoPermissao(modifier) { pedirPermissao.launch(Manifest.permission.CAMERA) }
    }
}

@Composable
private fun PedidoPermissao(modifier: Modifier, aoPermitir: () -> Unit) {
    Column(
        modifier = modifier
            .fillMaxSize()
            .padding(32.dp),
        verticalArrangement = Arrangement.Center,
        horizontalAlignment = Alignment.CenterHorizontally,
    ) {
        Box(
            contentAlignment = Alignment.Center,
            modifier = Modifier
                .size(72.dp)
                .background(MaterialTheme.colorScheme.primaryContainer, CircleShape),
        ) {
            Icon(
                painterResource(R.drawable.ic_camera),
                contentDescription = null,
                tint = MaterialTheme.colorScheme.primary,
                modifier = Modifier.size(32.dp),
            )
        }
        Spacer(Modifier.size(20.dp))
        Text("Permita o acesso à câmera", style = MaterialTheme.typography.titleMedium, fontWeight = FontWeight.Bold)
        Spacer(Modifier.size(8.dp))
        Text(
            "A câmera é usada para ler os códigos de barras ao vivo. Nenhuma imagem é salva.",
            style = MaterialTheme.typography.bodyMedium,
            color = MaterialTheme.colorScheme.onSurfaceVariant,
            textAlign = TextAlign.Center,
        )
        Spacer(Modifier.size(24.dp))
        Button(onClick = aoPermitir) { Text("Permitir câmera") }
    }
}

@Composable
private fun CameraComLeitura(verificador: VerificadorCodigos, detector: DetectorRfDetr, modifier: Modifier) {
    val context = LocalContext.current
    val lifecycleOwner = LocalLifecycleOwner.current
    val haptico = LocalHapticFeedback.current
    val escopo = rememberCoroutineScope()

    val consenso = remember { ConsensoFrames() }
    val situacoes = remember { mutableStateMapOf<String, SituacaoCodigo>() } // cache por valor
    val confirmados = remember { mutableStateListOf<CodigoLido>() } // mais recente primeiro
    var codigosNoFrame by remember { mutableStateOf(emptyList<CodigoLido>()) }
    var tamanhoPreview by remember { mutableStateOf(IntSize.Zero) }

    var temLanterna by remember { mutableStateOf(false) }
    var lanternaLigada by remember { mutableStateOf(false) }
    var analiseFoto by remember { mutableStateOf<AnaliseFoto>(AnaliseFoto.Ocioso) }
    val leitorFoto = remember(detector) { LeitorCodigos(detector = detector) }
    DisposableEffect(leitorFoto) { onDispose { leitorFoto.fechar() } }

    // Sugere a foto quando passa um tempo sem nenhuma confirmação.
    var instanteUltimaConfirmacao by remember { mutableLongStateOf(SystemClock.elapsedRealtime()) }
    var sugerirFoto by remember { mutableStateOf(false) }
    LaunchedEffect(instanteUltimaConfirmacao) {
        sugerirFoto = false
        delay(TEMPO_ATE_SUGERIR_FOTO_MS)
        sugerirFoto = true
    }

    fun consultarApi(codigo: CodigoLido) {
        if (codigo.valor in situacoes) return // já consultado: não chama a API de novo
        situacoes[codigo.valor] = SituacaoCodigo.CONSULTANDO
        confirmados.add(0, codigo)
        instanteUltimaConfirmacao = SystemClock.elapsedRealtime()
        haptico.performHapticFeedback(HapticFeedbackType.Confirm)
        escopo.launch {
            situacoes[codigo.valor] = try {
                if (verificador.existe(codigo)) SituacaoCodigo.EXISTE else SituacaoCodigo.NAO_ENCONTRADO
            } catch (erro: CancellationException) {
                throw erro
            } catch (erro: Exception) {
                SituacaoCodigo.ERRO
            }
        }
    }

    val controller = remember { LifecycleCameraController(context) }

    /** Fallback: tira a foto em resolução cheia e analisa no próprio aparelho. */
    fun analisarFoto() {
        if (analiseFoto == AnaliseFoto.Analisando) return
        // Marcado antes do launch: dois toques rápidos não disparam duas análises.
        analiseFoto = AnaliseFoto.Analisando
        escopo.launch {
            analiseFoto = try {
                val codigos = leitorFoto.ler(capturarFoto(controller)).codigos
                codigos.forEach { consultarApi(it.copy(origem = ORIGEM_FOTO)) }
                if (codigos.isEmpty()) AnaliseFoto.SemCodigos else AnaliseFoto.Ocioso
            } catch (erro: CancellationException) {
                throw erro
            } catch (erro: Exception) {
                AnaliseFoto.Falhou(erro.message ?: "Não foi possível analisar a foto.")
            }
        }
    }

    DisposableEffect(lifecycleOwner) {
        val executorPrincipal = ContextCompat.getMainExecutor(context)
        var scanner: BarcodeScanner? = null

        controller.imageAnalysisResolutionSelector = ResolutionSelector.Builder()
            .setResolutionStrategy(
                ResolutionStrategy(RESOLUCAO_ANALISE, ResolutionStrategy.FALLBACK_RULE_CLOSEST_HIGHER_THEN_LOWER)
            )
            .build()
        // A foto do fallback passa pelo RF-DETR e pelas 5 camadas: qualidade importa mais que rapidez.
        controller.imageCaptureMode = ImageCapture.CAPTURE_MODE_MAXIMIZE_QUALITY
        controller.imageCaptureResolutionSelector = ResolutionSelector.Builder()
            .setResolutionStrategy(ResolutionStrategy.HIGHEST_AVAILABLE_STRATEGY)
            .build()
        controller.bindToLifecycle(lifecycleOwner)

        // O zoom máximo da câmera só é conhecido depois que ela inicializa;
        // por isso o scanner (com zoom automático) é criado aqui.
        controller.initializationFuture.addListener({
            temLanterna = controller.cameraInfo?.hasFlashUnit() == true
            val novoScanner = criarScanner1D(opcoesZoomAutomatico(controller))
            scanner = novoScanner
            controller.setImageAnalysisAnalyzer(
                executorPrincipal,
                MlKitAnalyzer(listOf(novoScanner), COORDINATE_SYSTEM_VIEW_REFERENCED, executorPrincipal) { resultado ->
                    val mira = calcularMira(tamanhoPreview)
                    val lidos = resultado.getValue(novoScanner).orEmpty()
                        .mapNotNull { converterCodigo(it, ORIGEM_CAMERA) }
                        .filter { codigo -> codigo.caixa?.let(mira::contemCentroDe) == true }
                        .let(::filtrarCodigosConfirmados)
                    codigosNoFrame = lidos
                    consenso.registrar(lidos).forEach(::consultarApi)
                },
            )
        }, executorPrincipal)

        onDispose {
            controller.clearImageAnalysisAnalyzer()
            controller.unbind()
            scanner?.close()
        }
    }

    Column(modifier.fillMaxSize()) {
        Box(
            Modifier
                .fillMaxWidth()
                .weight(1f)
                .background(Color.Black)
                .onSizeChanged { tamanhoPreview = it },
        ) {
            AndroidView(
                factory = { contextoView ->
                    PreviewView(contextoView).apply {
                        implementationMode = PreviewView.ImplementationMode.COMPATIBLE
                        scaleType = PreviewView.ScaleType.FILL_CENTER
                        this.controller = controller
                    }
                },
                modifier = Modifier.fillMaxSize(),
            )
            SobreposicaoMira(tamanhoPreview, codigosNoFrame)
            val analisando = analiseFoto == AnaliseFoto.Analisando
            // A câmera segue lendo ao vivo enquanto a foto é analisada.
            AvisoStatus(
                detectando = codigosNoFrame.isNotEmpty(),
                analisandoFoto = analisando,
                modifier = Modifier
                    .align(Alignment.TopCenter)
                    .padding(top = 16.dp),
            )

            if (sugerirFoto && codigosNoFrame.isEmpty() && !analisando) {
                Button(
                    onClick = ::analisarFoto,
                    modifier = Modifier
                        .align(Alignment.BottomCenter)
                        .padding(bottom = 20.dp),
                ) {
                    Icon(painterResource(R.drawable.ic_camera), contentDescription = null, Modifier.size(18.dp))
                    Spacer(Modifier.width(8.dp))
                    Text("Não está lendo? Tirar foto")
                }
            }
            if (temLanterna) {
                BotaoLanterna(
                    ligada = lanternaLigada,
                    modifier = Modifier
                        .align(Alignment.BottomEnd)
                        .padding(end = 16.dp, bottom = 84.dp),
                ) {
                    lanternaLigada = !lanternaLigada
                    controller.enableTorch(lanternaLigada)
                }
            }
        }
        PainelLeituras(
            confirmados = confirmados.take(MAXIMO_CODIGOS_NA_LISTA),
            situacoes = situacoes,
            aviso = when (val analise = analiseFoto) {
                AnaliseFoto.SemCodigos -> "A análise da foto também não encontrou códigos. Aproxime a câmera ou melhore a luz."
                is AnaliseFoto.Falhou -> analise.mensagem
                else -> null
            },
            onLimpar = {
                consenso.limpar()
                situacoes.clear()
                confirmados.clear()
                analiseFoto = AnaliseFoto.Ocioso
                controller.setZoomRatio(1f) // desfaz o zoom automático
            },
        )
    }
}

/** Escurece tudo fora da mira, desenha os cantos e destaca os códigos detectados. */
@Composable
private fun SobreposicaoMira(tamanhoPreview: IntSize, codigosNoFrame: List<CodigoLido>) {
    val detectando = codigosNoFrame.isNotEmpty()
    val pulso by rememberInfiniteTransition(label = "linha de leitura").animateFloat(
        initialValue = 0.35f,
        targetValue = 1f,
        animationSpec = infiniteRepeatable(tween(durationMillis = 900), RepeatMode.Reverse),
        label = "opacidade da linha",
    )

    Canvas(Modifier.fillMaxSize()) {
        val mira = calcularMira(tamanhoPreview).paraRect()
        val raio = CornerRadius(16.dp.toPx())
        val recorte = Path().apply { addRoundRect(RoundRect(mira, raio)) }
        clipPath(recorte, ClipOp.Difference) {
            drawRect(Color.Black.copy(alpha = OPACIDADE_MASCARA))
        }

        val corMira = if (detectando) VerdeSucessoClaro else Color.White
        desenharCantos(mira, corMira)
        if (!detectando) {
            val margem = 20.dp.toPx()
            drawLine(
                color = VermelhoErro.copy(alpha = pulso),
                start = Offset(mira.left + margem, mira.center.y),
                end = Offset(mira.right - margem, mira.center.y),
                strokeWidth = 2.dp.toPx(),
                cap = StrokeCap.Round,
            )
        }

        codigosNoFrame.forEach { codigo ->
            val caixa = codigo.caixa?.paraRect() ?: return@forEach
            val raioCaixa = CornerRadius(6.dp.toPx())
            drawRoundRect(VerdeSucessoClaro.copy(alpha = 0.18f), caixa.topLeft, caixa.size, raioCaixa)
            drawRoundRect(VerdeSucessoClaro, caixa.topLeft, caixa.size, raioCaixa, Stroke(2.dp.toPx()))
        }
    }
}

private fun DrawScope.desenharCantos(mira: Rect, cor: Color) {
    val comprimento = 28.dp.toPx()
    val espessura = 4.dp.toPx()
    val cantos = listOf(
        Triple(mira.topLeft, 1f, 1f),
        Triple(mira.topRight, -1f, 1f),
        Triple(mira.bottomLeft, 1f, -1f),
        Triple(mira.bottomRight, -1f, -1f),
    )
    cantos.forEach { (ponto, sentidoX, sentidoY) ->
        drawLine(cor, ponto, ponto + Offset(comprimento * sentidoX, 0f), espessura, StrokeCap.Round)
        drawLine(cor, ponto, ponto + Offset(0f, comprimento * sentidoY), espessura, StrokeCap.Round)
    }
}

@Composable
private fun AvisoStatus(detectando: Boolean, analisandoFoto: Boolean, modifier: Modifier = Modifier) {
    val texto = when {
        analisandoFoto -> "Analisando a foto… continue apontando"
        detectando -> "Código detectado · segure firme"
        else -> "Posicione o código dentro da moldura"
    }
    Row(
        verticalAlignment = Alignment.CenterVertically,
        modifier = modifier
            .background(Color.Black.copy(alpha = 0.6f), RoundedCornerShape(50))
            .padding(horizontal = 16.dp, vertical = 8.dp),
    ) {
        if (analisandoFoto) {
            CircularProgressIndicator(Modifier.size(14.dp), color = Color.White, strokeWidth = 2.dp)
            Spacer(Modifier.width(8.dp))
        }
        Text(texto, style = MaterialTheme.typography.labelLarge, color = Color.White)
    }
}

@Composable
private fun BotaoLanterna(ligada: Boolean, modifier: Modifier = Modifier, aoAlternar: () -> Unit) {
    Box(
        contentAlignment = Alignment.Center,
        modifier = modifier
            .size(48.dp)
            .background(if (ligada) Color.White else Color.Black.copy(alpha = 0.6f), CircleShape)
            .clickable(onClickLabel = if (ligada) "Desligar lanterna" else "Ligar lanterna", onClick = aoAlternar),
    ) {
        Icon(
            painterResource(R.drawable.ic_lanterna),
            contentDescription = null,
            tint = if (ligada) Color.Black else Color.White,
        )
    }
}

@Composable
private fun PainelLeituras(
    confirmados: List<CodigoLido>,
    situacoes: Map<String, SituacaoCodigo>,
    aviso: String?,
    onLimpar: () -> Unit,
) {
    Surface(
        shape = RoundedCornerShape(topStart = 24.dp, topEnd = 24.dp),
        color = MaterialTheme.colorScheme.surface,
        shadowElevation = 8.dp,
        modifier = Modifier.fillMaxWidth(),
    ) {
        Column(
            Modifier
                .navigationBarsPadding()
                .padding(start = 20.dp, end = 12.dp, top = 4.dp, bottom = 8.dp),
        ) {
            Row(verticalAlignment = Alignment.CenterVertically) {
                Text("Leituras", style = MaterialTheme.typography.titleMedium, fontWeight = FontWeight.Bold)
                if (confirmados.isNotEmpty()) {
                    Spacer(Modifier.width(8.dp))
                    MarcadorNumero(confirmados.size)
                }
                Spacer(Modifier.weight(1f))
                TextButton(onClick = onLimpar, enabled = confirmados.isNotEmpty() || aviso != null) { Text("Limpar") }
            }

            aviso?.let {
                Text(
                    it,
                    style = MaterialTheme.typography.bodySmall,
                    color = MaterialTheme.colorScheme.error,
                    modifier = Modifier.padding(end = 8.dp, bottom = 4.dp),
                )
            }

            if (confirmados.isEmpty()) {
                Text(
                    "Os códigos confirmados aparecem aqui.",
                    style = MaterialTheme.typography.bodyMedium,
                    color = MaterialTheme.colorScheme.onSurfaceVariant,
                    modifier = Modifier.padding(end = 8.dp, bottom = 4.dp),
                )
            } else {
                Column(
                    Modifier
                        .heightIn(max = ALTURA_MAXIMA_LISTA)
                        .verticalScroll(rememberScrollState()),
                ) {
                    confirmados.forEachIndexed { indice, codigo ->
                        if (indice > 0) HorizontalDivider(color = MaterialTheme.colorScheme.surfaceVariant)
                        ItemLeitura(codigo, situacoes[codigo.valor] ?: SituacaoCodigo.CONSULTANDO)
                    }
                }
            }
        }
    }
}

@Composable
private fun ItemLeitura(codigo: CodigoLido, situacao: SituacaoCodigo) {
    Row(
        verticalAlignment = Alignment.CenterVertically,
        modifier = Modifier
            .fillMaxWidth()
            .padding(vertical = 8.dp)
            .padding(end = 8.dp),
    ) {
        Column(Modifier.weight(1f), verticalArrangement = Arrangement.spacedBy(4.dp)) {
            ValorCodigo(codigo.valor)
            Row(verticalAlignment = Alignment.CenterVertically) {
                EtiquetaFormato(codigo.formato)
                if (codigo.origem == ORIGEM_FOTO) {
                    Spacer(Modifier.width(6.dp))
                    Text("via foto", style = MaterialTheme.typography.labelSmall, color = MaterialTheme.colorScheme.primary)
                }
            }
        }
        Spacer(Modifier.width(12.dp))
        if (situacao == SituacaoCodigo.CONSULTANDO) {
            Row(verticalAlignment = Alignment.CenterVertically) {
                CircularProgressIndicator(Modifier.size(16.dp), strokeWidth = 2.dp)
                Spacer(Modifier.width(8.dp))
                Text(situacao.texto, style = MaterialTheme.typography.labelMedium, color = situacao.cor)
            }
        } else {
            SeloStatus(situacao.texto, situacao.cor)
        }
    }
}

/** Zoom automático do ML Kit; null se a câmera não tem zoom (ex.: webcam no emulador). */
private fun opcoesZoomAutomatico(controller: LifecycleCameraController): ZoomSuggestionOptions? {
    val zoomMaximoCamera = controller.cameraInfo?.zoomState?.value?.maxZoomRatio ?: return null
    if (zoomMaximoCamera <= 1f) return null
    return ZoomSuggestionOptions.Builder { zoom ->
        if (controller.cameraInfo == null) return@Builder false // câmera já fechada
        controller.setZoomRatio(zoom)
        true
    }
        .setMaxSupportedZoomRatio(minOf(zoomMaximoCamera, ZOOM_MAXIMO_AUTOMATICO))
        .build()
}

private fun calcularMira(tamanho: IntSize): Caixa {
    val alturaMira = (tamanho.height * MIRA_ALTURA).toInt()
    val topo = (tamanho.height - alturaMira) / 2
    return Caixa(
        esquerda = (tamanho.width * MIRA_MARGEM_LATERAL).toInt(),
        topo = topo,
        direita = (tamanho.width * (1 - MIRA_MARGEM_LATERAL)).toInt(),
        base = topo + alturaMira,
    )
}

/** Tira uma foto e devolve já na orientação certa e no tamanho limite do app. */
private suspend fun capturarFoto(controller: LifecycleCameraController): Bitmap =
    suspendCancellableCoroutine { continuacao ->
        controller.takePicture(
            Dispatchers.Default.asExecutor(),
            object : ImageCapture.OnImageCapturedCallback() {
                override fun onCaptureSuccess(imagem: ImageProxy) {
                    try {
                        continuacao.resume(endireitar(imagem))
                    } catch (erro: Exception) {
                        continuacao.resumeWithException(erro)
                    } finally {
                        imagem.close()
                    }
                }

                override fun onError(erro: ImageCaptureException) {
                    continuacao.resumeWithException(erro)
                }
            },
        )
    }

private fun endireitar(imagem: ImageProxy): Bitmap {
    val rotacao = imagem.imageInfo.rotationDegrees
    val original = imagem.toBitmap()
    val endireitada = if (rotacao == 0) original else Bitmap.createBitmap(
        original, 0, 0, original.width, original.height,
        Matrix().apply { postRotate(rotacao.toFloat()) }, true,
    )
    return limitarTamanho(endireitada)
}

private fun Caixa.paraRect() = Rect(
    offset = Offset(esquerda.toFloat(), topo.toFloat()),
    size = TamanhoCompose(largura.toFloat(), altura.toFloat()),
)
