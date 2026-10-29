package io.kotensor.sample.ui

import androidx.compose.foundation.Canvas
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.verticalScroll
import androidx.compose.material3.Button
import androidx.compose.material3.Card
import androidx.compose.material3.LinearProgressIndicator
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.OutlinedButton
import androidx.compose.material3.OutlinedTextField
import androidx.compose.material3.Scaffold
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.collectAsState
import androidx.compose.runtime.getValue
import androidx.compose.ui.Modifier
import androidx.compose.ui.geometry.Offset
import androidx.compose.ui.unit.dp
import androidx.lifecycle.viewmodel.compose.viewModel
import io.kotensor.core.Quantization
import io.kotensor.sample.EngineViewModel
import io.kotensor.sample.MemoryTelemetry
import io.kotensor.sample.UiState
import java.util.Locale

@Composable
fun KotensorScreen(viewModel: EngineViewModel = viewModel()) {
    val state by viewModel.state.collectAsState()
    MaterialTheme {
        Scaffold { padding ->
            Column(
                modifier = Modifier
                    .fillMaxSize()
                    .padding(padding)
                    .verticalScroll(rememberScrollState())
                    .padding(16.dp),
                verticalArrangement = Arrangement.spacedBy(12.dp)
            ) {
                Text("Kotensor", style = MaterialTheme.typography.headlineMedium)
                StatusCard(state, onReload = viewModel::loadModel)
                TelemetryCard(state.telemetry)
                GenerationCard(state, onPrompt = viewModel::setPrompt, onGenerate = viewModel::generate, onCancel = viewModel::cancel)
                FineTuneCard(state, onText = viewModel::setFineTuneText, onFineTune = viewModel::fineTune)
                QuantizationCard(state, onQuantize = viewModel::quantize)
            }
        }
    }
}

@Composable
private fun StatusCard(state: UiState, onReload: () -> Unit) {
    Card(modifier = Modifier.fillMaxWidth()) {
        Column(modifier = Modifier.padding(16.dp), verticalArrangement = Arrangement.spacedBy(8.dp)) {
            Text(state.status, style = MaterialTheme.typography.titleMedium)
            if (state.modelDescription.isNotEmpty()) {
                Text(state.modelDescription, style = MaterialTheme.typography.bodyMedium)
            }
            if (state.busy) {
                LinearProgressIndicator(progress = { state.progress }, modifier = Modifier.fillMaxWidth())
            }
            if (state.trainingLosses.isNotEmpty()) {
                LossChart(state.trainingLosses, label = "demo training loss")
            }
            Text("Copy a Hugging Face model directory (config.json, model.safetensors, tokenizer.json) into files/model to load it as INT8 instead of the demo model, or a kotensor-quantized.safetensors with its tokenizer.json.", style = MaterialTheme.typography.bodySmall)
            OutlinedButton(onClick = onReload, enabled = !state.busy) { Text("Reload model") }
        }
    }
}

@Composable
private fun TelemetryCard(telemetry: MemoryTelemetry) {
    Card(modifier = Modifier.fillMaxWidth()) {
        Column(modifier = Modifier.padding(16.dp), verticalArrangement = Arrangement.spacedBy(4.dp)) {
            Text("Memory", style = MaterialTheme.typography.titleMedium)
            Text(format("Java heap %.1f / %.1f MB", telemetry.heapUsedMb, telemetry.heapMaxMb))
            Text(format("Native heap %.1f MB", telemetry.nativeHeapMb))
            Text(format("Model weights %.1f MB (%,d parameters)", telemetry.modelWeightsMb, telemetry.parameters))
            Text(format("KV cache for %d positions %.1f MB", telemetry.cachePositions, telemetry.cacheMb))
        }
    }
}

@Composable
private fun GenerationCard(state: UiState, onPrompt: (String) -> Unit, onGenerate: () -> Unit, onCancel: () -> Unit) {
    Card(modifier = Modifier.fillMaxWidth()) {
        Column(modifier = Modifier.padding(16.dp), verticalArrangement = Arrangement.spacedBy(8.dp)) {
            Text("Streaming generation", style = MaterialTheme.typography.titleMedium)
            OutlinedTextField(
                value = state.prompt,
                onValueChange = onPrompt,
                label = { Text("Prompt") },
                modifier = Modifier.fillMaxWidth(),
                enabled = !state.busy
            )
            Row(horizontalArrangement = Arrangement.spacedBy(8.dp)) {
                Button(onClick = onGenerate, enabled = state.ready && !state.busy) { Text("Generate") }
                OutlinedButton(onClick = onCancel, enabled = state.busy) { Text("Stop") }
            }
            Text(format("%d tokens · %.1f tokens/s", state.tokensGenerated, state.tokensPerSecond), style = MaterialTheme.typography.bodySmall)
            Text(state.prompt + state.output, style = MaterialTheme.typography.bodyLarge)
        }
    }
}

@Composable
private fun FineTuneCard(state: UiState, onText: (String) -> Unit, onFineTune: () -> Unit) {
    Card(modifier = Modifier.fillMaxWidth()) {
        Column(modifier = Modifier.padding(16.dp), verticalArrangement = Arrangement.spacedBy(8.dp)) {
            Text(if (state.quantization == "fp32") "LoRA fine-tuning" else "QLoRA fine-tuning on ${state.quantization} weights", style = MaterialTheme.typography.titleMedium)
            OutlinedTextField(
                value = state.fineTuneText,
                onValueChange = onText,
                label = { Text("Text to learn") },
                modifier = Modifier.fillMaxWidth(),
                minLines = 3,
                enabled = !state.busy
            )
            Button(onClick = onFineTune, enabled = state.ready && !state.busy) {
                Text(if (state.adapterAttached) "Fine-tune more" else "Attach adapter and fine-tune")
            }
            if (state.fineTuneLosses.isNotEmpty()) {
                Text(format("loss %.3f → %.3f", state.fineTuneLosses.first(), state.fineTuneLosses.last()), style = MaterialTheme.typography.bodySmall)
                LossChart(state.fineTuneLosses, label = "fine-tuning loss")
            }
        }
    }
}

@Composable
private fun QuantizationCard(state: UiState, onQuantize: (Quantization) -> Unit) {
    Card(modifier = Modifier.fillMaxWidth()) {
        Column(modifier = Modifier.padding(16.dp), verticalArrangement = Arrangement.spacedBy(8.dp)) {
            Text("Weights: ${state.quantization}", style = MaterialTheme.typography.titleMedium)
            Row(horizontalArrangement = Arrangement.spacedBy(8.dp)) {
                Button(onClick = { onQuantize(Quantization.INT8) }, enabled = state.ready && !state.busy && state.quantization == "fp32") { Text("Quantize INT8") }
                Button(onClick = { onQuantize(Quantization.INT4) }, enabled = state.ready && !state.busy && state.quantization == "fp32") { Text("Quantize INT4") }
            }
            Text("Quantization is in place: generation keeps streaming from the compact weights and fine-tuning switches to QLoRA.", style = MaterialTheme.typography.bodySmall)
        }
    }
}

@Composable
private fun LossChart(losses: List<Float>, label: String) {
    val color = MaterialTheme.colorScheme.primary
    val grid = MaterialTheme.colorScheme.outlineVariant
    Column {
        Text(label, style = MaterialTheme.typography.labelSmall)
        Spacer(modifier = Modifier.height(4.dp))
        Canvas(modifier = Modifier.fillMaxWidth().height(80.dp)) {
            val max = losses.max()
            val min = losses.min()
            val range = if (max > min) max - min else 1f
            drawLine(grid, Offset(0f, size.height), Offset(size.width, size.height), strokeWidth = 2f)
            if (losses.size < 2) return@Canvas
            val step = size.width / (losses.size - 1)
            for (i in 1 until losses.size) {
                val from = Offset((i - 1) * step, size.height - (losses[i - 1] - min) / range * size.height)
                val to = Offset(i * step, size.height - (losses[i] - min) / range * size.height)
                drawLine(color, from, to, strokeWidth = 4f)
            }
        }
    }
}

private fun format(pattern: String, vararg args: Any): String = String.format(Locale.US, pattern, *args)
