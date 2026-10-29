package io.kotensor.sample

import android.app.Application
import androidx.lifecycle.AndroidViewModel
import androidx.lifecycle.viewModelScope
import io.kotensor.core.AdamW
import io.kotensor.core.DataLoader
import io.kotensor.core.FineTuningPipeline
import io.kotensor.core.LoRAAdapter
import io.kotensor.core.Quantization
import io.kotensor.core.StreamingSession
import java.io.File
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.delay
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.flow.update
import kotlinx.coroutines.isActive
import kotlinx.coroutines.launch

data class UiState(
    val status: String = "Starting",
    val modelDescription: String = "",
    val ready: Boolean = false,
    val busy: Boolean = false,
    val progress: Float = 0f,
    val trainingLosses: List<Float> = emptyList(),
    val prompt: String = "the engine",
    val output: String = "",
    val tokensGenerated: Int = 0,
    val tokensPerSecond: Float = 0f,
    val fineTuneText: String = "the engine dreams in kotlin. the engine dreams in kotlin. the engine dreams in kotlin.",
    val fineTuneLosses: List<Float> = emptyList(),
    val adapterAttached: Boolean = false,
    val quantization: String = "fp32",
    val telemetry: MemoryTelemetry = MemoryTelemetry()
)

class EngineViewModel(application: Application) : AndroidViewModel(application) {
    private val _state = MutableStateFlow(UiState())
    val state: StateFlow<UiState> = _state.asStateFlow()

    private var loaded: LoadedModel? = null
    private var session: StreamingSession? = null
    private var adapter: LoRAAdapter? = null

    init {
        viewModelScope.launch(Dispatchers.Default) {
            while (isActive) {
                _state.update { it.copy(telemetry = Telemetry.sample(loaded?.model, CACHE_POSITIONS)) }
                delay(1000)
            }
        }
        loadModel()
    }

    fun setPrompt(prompt: String) = _state.update { it.copy(prompt = prompt) }

    fun setFineTuneText(text: String) = _state.update { it.copy(fineTuneText = text) }

    fun loadModel() = launchBusy("Loading model") {
        val directory = File(getApplication<Application>().filesDir, "model")
        val fromFiles = ModelSource.fromFiles(directory, ModelSource.IMPORT_QUANTIZATION) { name -> setStatus("Loading $name") }
        val result = fromFiles ?: ModelSource.trainDemo { step, total, loss ->
            _state.update { it.copy(status = "Training demo model $step/$total", progress = step.toFloat() / total, trainingLosses = it.trainingLosses + loss) }
        }
        loaded = result
        adapter = null
        session = StreamingSession(result.model, result.tokenizer)
        _state.update {
            it.copy(
                ready = true,
                modelDescription = result.description,
                quantization = result.model.quantization?.toString() ?: "fp32",
                adapterAttached = false,
                output = "",
                fineTuneLosses = emptyList()
            )
        }
        null
    }

    fun generate() = launchBusy("Generating") {
        val model = loaded ?: return@launchBusy NO_MODEL
        val streaming = session ?: return@launchBusy NO_MODEL
        _state.update { it.copy(output = "", tokensGenerated = 0, tokensPerSecond = 0f) }
        val prompt = _state.value.prompt
        streaming.stream(prompt, MAX_TOKENS, model.stopTokens) { generation ->
            _state.update { it.copy(output = generation.generatedText, tokensGenerated = generation.tokensGenerated, tokensPerSecond = generation.tokensPerSecond) }
        }
        null
    }

    fun cancel() {
        session?.cancel()
    }

    fun fineTune() = launchBusy("Fine-tuning with LoRA") {
        val model = loaded ?: return@launchBusy NO_MODEL
        val text = _state.value.fineTuneText
        val tokens = model.tokenizer.encode(text)
        if (tokens.size < 4) {
            return@launchBusy "Need more text to fine-tune"
        }
        val attached = adapter ?: LoRAAdapter(model.model, rank = 4, alpha = 2.0f).also { adapter = it }
        val optimizer = AdamW(attached.parameters(), learningRate = 2e-3f)
        val pipeline = FineTuningPipeline(model.model, attached, optimizer)
        val sequenceLength = minOf(32, tokens.size - 1)
        val loader = DataLoader(tokens, batchSize = 1, sequenceLength = sequenceLength)
        _state.update { it.copy(fineTuneLosses = emptyList(), adapterAttached = true) }
        pipeline.train(loader, epochs = FINE_TUNE_EPOCHS) { epoch, loss ->
            _state.update { it.copy(status = "Fine-tuning epoch $epoch/$FINE_TUNE_EPOCHS", progress = epoch.toFloat() / FINE_TUNE_EPOCHS, fineTuneLosses = it.fineTuneLosses + loss) }
        }
        null
    }

    fun quantize(quantization: Quantization) = launchBusy("Quantizing to $quantization") {
        val model = loaded ?: return@launchBusy NO_MODEL
        if (adapter != null) {
            return@launchBusy "Reload the model before quantizing: an adapter is attached"
        }
        model.model.quantize(quantization)
        _state.update { it.copy(quantization = quantization.toString()) }
        null
    }

    private fun setStatus(status: String) = _state.update { it.copy(status = status) }

    private fun launchBusy(label: String, block: suspend () -> String?) {
        if (_state.value.busy) return
        viewModelScope.launch(Dispatchers.Default) {
            _state.update { it.copy(busy = true, status = label, progress = 0f) }
            try {
                val message = block()
                _state.update { it.copy(status = message ?: "Ready", progress = 1f) }
            } catch (e: Exception) {
                _state.update { it.copy(status = "Error: ${e.message}") }
            } finally {
                _state.update { it.copy(busy = false) }
            }
        }
    }

    companion object {
        const val MAX_TOKENS = 160
        const val FINE_TUNE_EPOCHS = 20
        const val CACHE_POSITIONS = 2048
        const val NO_MODEL = "No model loaded"
    }
}
