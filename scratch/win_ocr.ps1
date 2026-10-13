Add-Type -AssemblyName System.Runtime.WindowsRuntime
$null = [Windows.Storage.StorageFile, Windows.Storage, ContentType=WindowsRuntime]
$null = [Windows.Graphics.Imaging.BitmapDecoder, Windows.Graphics.Imaging, ContentType=WindowsRuntime]
$null = [Windows.Media.Ocr.OcrEngine, Windows.Foundation, ContentType=WindowsRuntime]

function AwaitTask($task) {
    $asTaskGeneric = [System.WindowsRuntimeSystemExtensions].GetMethods() | Where-Object { $_.Name -eq 'AsTask' -and $_.GetParameters().Count -eq 1 -and $_.GetParameters()[0].ParameterType.Name -eq 'IAsyncOperation`1' }
    $netTask = $asTaskGeneric[0].MakeGenericMethod($task.GetType().GetGenericArguments()[0]).Invoke($null, @($task))
    $netTask.Wait(-1) | Out-Null
    return $netTask.Result
}

$imgPath = (Resolve-Path 'scratch/exact_row.png').Path
$file = AwaitTask ([Windows.Storage.StorageFile]::GetFileFromPathAsync($imgPath))
$stream = AwaitTask ($file.OpenAsync([Windows.Storage.FileAccessMode]::Read))
$decoder = AwaitTask ([Windows.Graphics.Imaging.BitmapDecoder]::CreateAsync($stream))
$bitmap = AwaitTask ($decoder.GetSoftwareBitmapAsync())

$engine = [Windows.Media.Ocr.OcrEngine]::TryCreateFromLanguage([Windows.Globalization.Language]::new('ko'))
$ocrResult = AwaitTask ($engine.RecognizeAsync($bitmap))

Write-Output "--- Windows Media OCR Lines ---"
foreach ($line in $ocrResult.Lines) {
    Write-Output $line.Text
}
