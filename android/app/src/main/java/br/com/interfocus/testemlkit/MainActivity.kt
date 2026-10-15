package br.com.interfocus.testemlkit

import android.os.Bundle
import androidx.activity.ComponentActivity
import androidx.activity.compose.setContent
import androidx.activity.enableEdgeToEdge
import androidx.annotation.DrawableRes
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.WindowInsets
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.statusBarsPadding
import androidx.compose.material3.Icon
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.PrimaryTabRow
import androidx.compose.material3.Scaffold
import androidx.compose.material3.Surface
import androidx.compose.material3.Tab
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.DisposableEffect
import androidx.compose.runtime.LaunchedEffect
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableIntStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.saveable.rememberSaveable
import androidx.compose.runtime.setValue
import androidx.compose.ui.Modifier
import androidx.compose.ui.res.painterResource
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.dp
import br.com.interfocus.testemlkit.ui.theme.TesteMlKitTheme
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext

private enum class Aba(val titulo: String, @DrawableRes val icone: Int) {
    CAMERA("Câmera ao vivo", R.drawable.ic_camera),
    FOTO("Foto", R.drawable.ic_imagem),
}

class MainActivity : ComponentActivity() {
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        enableEdgeToEdge()
        setContent {
            TesteMlKitTheme {
                val verificador = remember { VerificadorSimulado() }
                // Um só RF-DETR para as duas abas: o modelo ocupa bastante memória.
                val detector = remember { DetectorRfDetr(applicationContext) }
                DisposableEffect(detector) { onDispose { detector.close() } }
                // Se falhar aqui, a falha reaparece (e é mostrada) na primeira foto analisada.
                LaunchedEffect(detector) { withContext(Dispatchers.Default) { runCatching { detector.preparar() } } }
                var abaSelecionada by rememberSaveable { mutableIntStateOf(Aba.CAMERA.ordinal) }

                Scaffold(
                    modifier = Modifier.fillMaxSize(),
                    topBar = { Cabecalho(abaSelecionada) { abaSelecionada = it } },
                    // Cada tela trata a barra de navegação do sistema (o painel da câmera vai até a borda).
                    contentWindowInsets = WindowInsets(0),
                ) { espacamento ->
                    val conteudo = Modifier.padding(espacamento)
                    when (Aba.entries[abaSelecionada]) {
                        Aba.CAMERA -> TelaCamera(verificador, detector, conteudo)
                        Aba.FOTO -> TelaLeitura(detector, conteudo)
                    }
                }
            }
        }
    }
}

@Composable
private fun Cabecalho(abaSelecionada: Int, aoSelecionar: (Int) -> Unit) {
    Surface(color = MaterialTheme.colorScheme.surface, shadowElevation = 2.dp) {
        Column(Modifier.statusBarsPadding()) {
            Column(Modifier.padding(start = 20.dp, end = 20.dp, top = 12.dp, bottom = 4.dp)) {
                Text(
                    text = "Leitor de códigos",
                    style = MaterialTheme.typography.titleLarge,
                    fontWeight = FontWeight.Bold,
                )
                Text(
                    text = "Códigos de barras de equipamentos",
                    style = MaterialTheme.typography.bodySmall,
                    color = MaterialTheme.colorScheme.onSurfaceVariant,
                )
            }
            PrimaryTabRow(
                selectedTabIndex = abaSelecionada,
                containerColor = MaterialTheme.colorScheme.surface,
            ) {
                Aba.entries.forEach { aba ->
                    Tab(
                        selected = abaSelecionada == aba.ordinal,
                        onClick = { aoSelecionar(aba.ordinal) },
                        text = { Text(aba.titulo) },
                        icon = { Icon(painterResource(aba.icone), contentDescription = null) },
                        unselectedContentColor = MaterialTheme.colorScheme.onSurfaceVariant,
                    )
                }
            }
        }
    }
}
