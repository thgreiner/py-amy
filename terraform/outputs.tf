output "address" {
  value = "${aws_elb.web.dns_name}"
}

output "public_ip" {
  value = "${aws_spot_instance_request.web.public_ip}"
}