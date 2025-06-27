<?php
$conn = new mysqli("localhost", "webuser", "webpassword", "sql_injection_demo");
if ($conn->connect_error) {
    die("Connection failed: " . $conn->connect_error);
}
if ($_SERVER["REQUEST_METHOD"] == "POST") {
    $username = $_POST["username"];
    $password = $_POST["password"];
    $sql = "SELECT * FROM users WHERE username = '$username' AND password = '$p>
    $result = $conn->query($sql);
    if ($result->num_rows > 0) {
        echo "Đăng nhập thành công!<br>";
        while ($row = $result->fetch_assoc()) {
            echo "Username: " . $row["username"] . " - Password: " . $row["pass>
        }
    } else {
        echo "Sai tên đăng nhập hoặc mật khẩu.";
    }
}
?>