package br.com.interfocus.testemlkit

import androidx.compose.foundation.background
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.size
import androidx.compose.foundation.shape.CircleShape
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.text.font.FontFamily
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.text.style.TextOverflow
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp

/* Peças visuais usadas tanto na câmera ao vivo quanto na leitura por foto. */

/** Pílula com o nome do formato (ex.: "Code 128"). */
@Composable
fun EtiquetaFormato(formato: FormatoCodigo, modifier: Modifier = Modifier) {
    Text(
        text = formato.nome,
        style = MaterialTheme.typography.labelSmall,
        color = MaterialTheme.colorScheme.onSecondaryContainer,
        modifier = modifier
            .background(MaterialTheme.colorScheme.secondaryContainer, RoundedCornerShape(50))
            .padding(horizontal = 8.dp, vertical = 2.dp),
    )
}

/** Pílula colorida de status (ex.: "EXISTE" em verde). */
@Composable
fun SeloStatus(texto: String, cor: Color, modifier: Modifier = Modifier) {
    Text(
        text = texto,
        style = MaterialTheme.typography.labelMedium,
        fontWeight = FontWeight.SemiBold,
        color = cor,
        modifier = modifier
            .background(cor.copy(alpha = 0.12f), RoundedCornerShape(50))
            .padding(horizontal = 10.dp, vertical = 4.dp),
    )
}

/** Valor do código em fonte monoespaçada, que facilita conferir caractere a caractere. */
@Composable
fun ValorCodigo(valor: String, modifier: Modifier = Modifier) {
    Text(
        text = valor,
        style = MaterialTheme.typography.bodyLarge,
        fontFamily = FontFamily.Monospace,
        fontWeight = FontWeight.Medium,
        maxLines = 1,
        overflow = TextOverflow.Ellipsis,
        modifier = modifier,
    )
}

/** Círculo numerado; o mesmo número aparece sobre o código na imagem. */
@Composable
fun MarcadorNumero(numero: Int, modifier: Modifier = Modifier) {
    Box(
        contentAlignment = Alignment.Center,
        modifier = modifier
            .size(28.dp)
            .background(MaterialTheme.colorScheme.primary, CircleShape),
    ) {
        Text(
            text = numero.toString(),
            color = MaterialTheme.colorScheme.onPrimary,
            fontSize = 13.sp,
            fontWeight = FontWeight.Bold,
        )
    }
}
