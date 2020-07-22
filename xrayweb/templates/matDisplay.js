function toFormula(name) {
    var i = keys.indexOf(name);
    if (i != -1) {
        return vals[i][0];
    }
    return "";
}
function toDensity(name) {
    var i = keys.indexOf(name);
    if (i != -1) {
        return vals[i][1];
    }
    return "";
}
function displayinfo() {
    var dropdown = document.getElementById('mats');
    document.getElementById('formula').value=toFormula(dropdown.options[dropdown.selectedIndex].text);
    document.getElementById('density').value=toDensity(dropdown.options[dropdown.selectedIndex].text);
}