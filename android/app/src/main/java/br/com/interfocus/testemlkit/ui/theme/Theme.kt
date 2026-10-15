package br.com.interfocus.testemlkit.ui.theme

import androidx.compose.foundation.isSystemInDarkTheme
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.darkColorScheme
import androidx.compose.material3.lightColorScheme
import androidx.compose.runtime.Composable
import androidx.compose.ui.graphics.Color

private val EsquemaClaro = lightColorScheme(
    primary = Azul600,
    onPrimary = Color.White,
    primaryContainer = Azul100,
    onPrimaryContainer = Azul900,
    secondary = Cinza700,
    onSecondary = Color.White,
    secondaryContainer = Cinza100,
    onSecondaryContainer = Cinza900,
    background = Cinza50,
    onBackground = Cinza900,
    surface = Color.White,
    onSurface = Cinza900,
    surfaceVariant = Cinza100,
    onSurfaceVariant = Cinza700,
    outline = Cinza400,
    error = VermelhoErro,
)

private val EsquemaEscuro = darkColorScheme(
    primary = Azul200,
    onPrimary = Azul900,
    primaryContainer = Azul600,
    onPrimaryContainer = Color.White,
    secondary = Cinza400,
    onSecondary = Cinza900,
    secondaryContainer = Grafite2,
    onSecondaryContainer = Cinza100,
    background = Cinza900,
    onBackground = Cinza100,
    surface = Grafite,
    onSurface = Cinza100,
    surfaceVariant = Grafite2,
    onSurfaceVariant = Cinza400,
    outline = Cinza700,
    error = Color(0xFFFF8A80),
)

/**
 * Tema fixo do app. As cores dinâmicas do Android 12+ ficam desligadas para a
 * identidade visual ser a mesma em qualquer aparelho.
 */
@Composable
fun TesteMlKitTheme(
    darkTheme: Boolean = isSystemInDarkTheme(),
    content: @Composable () -> Unit,
) {
    MaterialTheme(
        colorScheme = if (darkTheme) EsquemaEscuro else EsquemaClaro,
        typography = Typography,
        content = content,
    )
}
